/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <time.h>
#include <map>
#include <cctype>
#include <chrono>
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // Set the number of particles. Initialize all particles to first position (based on estimates of
    // x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine gen(seed);

    normal_distribution<double> x_dist(x, std[0]);
    normal_distribution<double> y_dist(x, std[1]);
    normal_distribution<double> theta_dist(theta, std[2]);

    num_particles = 100;
    weights.resize(num_particles);

    for (int i=0; i<num_particles; i++) {
        Particle p = {
            i,                  //id
            x_dist(gen),        //x
            y_dist(gen),        //y
            theta_dist(gen),    //theta
            1.0                 //weight
        };
        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // Use the bicycle model to predict the next location of the particle with added random Gaussian noise.

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine gen(seed);

    for (auto &p: particles) {
        if (fabs(yaw_rate) < 0.00001) {
            p.x += velocity*delta_t*cos(p.theta);
            p.y += velocity*delta_t*sin(p.theta);
        } else {
            p.x += (velocity / yaw_rate) * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
            p.y += (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
            p.theta += yaw_rate*delta_t;
        }

        normal_distribution<double> dist_x(p.x, std_pos[0]);
        normal_distribution<double> dist_y(p.y, std_pos[1]);
        normal_distribution<double> dist_theta(p.theta, std_pos[2]);

        // set noisy prediction
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

    // performed as part of update step
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

    // Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    // more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    //
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
    //   for the fact that the map's y-axis actually points downwards.)
    //   http://planning.cs.uiuc.edu/node99.html

    for (int i=0; i<num_particles; i++) {
        Particle &p = particles[i];

        double weight = 1.0;

        for (int j=0; j<observations.size(); j++) {
            LandmarkObs obs = observations[j];

            // Convert observation to world/map coordinates
            double obs_x, obs_y;
            obs_x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
            obs_y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);

            // Calculate the distances from the observation to each landmark
            // and cache the closest landmark. This is just a basic nearest
            // neighbour algorithm.
            Map::single_landmark_s closest_lm = {0, 0.0, 0.0};
            double min_distance_obs_to_landmark = 1000.0;

            for (int k=0; k<map_landmarks.landmark_list.size(); k++) {
                Map::single_landmark_s landmark = map_landmarks.landmark_list[k];
                double landmark_x = landmark.x_f;
                double landmark_y = landmark.y_f;

                //double distance_obs_to_landmark = dist(obs_x, obs_y, map_x, map_y);
                double distance_particle_to_landmark = dist(p.x, p.y, landmark_x, landmark_y);

                if (distance_particle_to_landmark <= sensor_range) {
                    double distance_obs_to_landmark = dist(obs_x, obs_y, landmark_x, landmark_y);

                    if (distance_obs_to_landmark < min_distance_obs_to_landmark) {
                        min_distance_obs_to_landmark = distance_obs_to_landmark;
                        closest_lm = landmark;
                    }
                }
            }

            // Calculate bivariate gaussian - this belongs in helper_functions.cpp
            // however for Udacity SDC grading the code needs to be inlined here...
            //weight *= bivariate_gaussian(closest_lm, obs_x, obs_y, std_landmark);

            double map_x = closest_lm.x_f;
            double map_y = closest_lm.y_f;
            double x_range = map_x - obs_x;
            double y_range = map_y - obs_y;
            double std_x = std_landmark[0];
            double std_y = std_landmark[1];

            double x_y_term = ((x_range*x_range)/(std_x*std_x)) + ((y_range*y_range)/(std_y*std_y));
            long double w = exp(-0.5*x_y_term) / (2*M_PI*std_x*std_y);

            if (w < 0.0001) {
                w = 0.0001;
            }
            weight *= w;
        }

        p.weight = weight;
        weights[i] = weight;
    }

    // Normalise the particle weights
    double weights_sum = accumulate(weights.begin(), weights.end(), 0.0);
    for (int i=0; i<num_particles; i++) {
        Particle &p = particles[i];
        p.weight = p.weight / weights_sum;
        weights[i] = p.weight;
    }

}

void ParticleFilter::resample() {
    // Resample particles with replacement with probability proportional to their weight.
    //
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    //
    // C++ implementation of Sebastions python wheel-resampling logic - Particle Filters Lesson 20.

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine gen(seed);

    std::discrete_distribution<> d(weights.begin(), weights.end());
    vector<Particle> resampled_particles;

    for (int i=0; i<num_particles; i++) {
        int index = d(gen);
        resampled_particles.push_back(particles[index]);
    }

    particles = resampled_particles;

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
