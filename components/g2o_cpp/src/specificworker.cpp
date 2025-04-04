/*
 *    Copyright (C) 2024 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "specificworker.h"

/**
* \brief Default constructor
*/
SpecificWorker::SpecificWorker(TuplePrx tprx, bool startup_check) : GenericWorker(tprx)
{
	this->startup_check_flag = startup_check;
}

/**
* \brief Default destructor
*/
SpecificWorker::~SpecificWorker()
{
	std::cout << "Destroying SpecificWorker" << std::endl;
}

bool SpecificWorker::setParams(RoboCompCommonBehavior::ParameterList params)
{
    parameters.iterations =std::stoi(params.at("Iterations").value);
	return true;
}

void SpecificWorker::initialize(int period)
{
	std::cout << "Initialize worker" << std::endl;

	this->Period = 2000;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
        // Initialize the optimizer
        optimizer.setVerbose(true);

        // Linear solver
        linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
        linear_solver->setBlockOrdering(false);

        // Block solver
        block_solver = std::make_unique<g2o::BlockSolverX>(std::move(linear_solver));

        // Optimization algorithm
        algorithm = std::make_unique<g2o::OptimizationAlgorithmLevenberg>(std::move(block_solver));
        optimizer.setAlgorithm(algorithm.get());

        // Set the delta parameter
        robustKernel->setDelta(1.0);

        timer.start(Period);
	}
}

void SpecificWorker::compute()
{
    // Print alive message
    fps.print("FPS:", 3000);
}

////////////////////////////////////////////////////////////////////////////////////////////
int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}


std::string SpecificWorker::G2Ooptimizer_optimize(std::string trajectory)
{

    std::setlocale(LC_NUMERIC, "C");

    // Save temporary file with the trajectory
    std::ofstream file("trajectory.g2o");

    if (!file.is_open())
    {
        std::cerr << "Failed to open file for writing." << std::endl;
        return "";
    }
    file << trajectory;
    file.close();

    optimizer.clear();

    if (!optimizer.load("trajectory.g2o"))
    {
        qWarning() << "Error loading graph from file: " << QString::fromStdString("trajectory.g2o");
        qWarning() << "Returning empty string.";
        return "";
    }

    //TODO: fix segmentation fault using RobustKernel (SE2_ROBUST)
//    // Iterate over all edges in the graph
//    for (g2o::OptimizableGraph::EdgeSet::iterator it = optimizer.edges().begin(); it != optimizer.edges().end(); ++it) {
//        g2o::OptimizableGraph::Edge* edge = static_cast<g2o::OptimizableGraph::Edge*>(*it);
//        // Set the robust kernel for the current edge
//        edge->setRobustKernel(robustKernel);
//    }

    // Perform optimization
    optimizer.initializeOptimization();
    optimizer.optimize(parameters.iterations);

    // Covariances. Assuming the landmarks/vertices of interest are the first four vertices
    // Open a file to write the covariance matrices
    std::ofstream file_cov("covariances.txt");
    if (!file_cov.is_open())
    {
        std::cerr << "Failed to open file for writing." << std::endl;
    }
    else
    {
        for (int i = 0; i < 4; ++i)
        {
            g2o::OptimizableGraph::Vertex *v = optimizer.vertex(i);
            if (v)
            {
                g2o::SparseBlockMatrix<Eigen::MatrixXd> spinv;
                bool state = optimizer.computeMarginals(spinv, v);
                if(state)
                {
                    g2o::SparseBlockMatrix<Eigen::MatrixXd>::SparseMatrixBlock b = spinv.block(i, i)->eval();
                    auto inverse_b = b;
                    std::cout << "Covariance matrix for corner " << i << std::endl;
                    std::cout << inverse_b << std::endl;
                    // Write the vertex ID and the covariance matrix
                    file_cov << "Vertex " << i << " Covariance Matrix:\n";
                    file_cov << inverse_b(0, 0) << " " << inverse_b(0, 1) << "\n";
                    file_cov << inverse_b(1, 0) << " " << inverse_b(1, 1) << "\n";
                    file_cov << "\n";  // Add a blank line between entries for readability
                }
            }
        }
        file.close();
    }

    // Save the optimized graph
    optimizer.save("optimized_trajectory.g2o");
    std::cout << "Optimization complete. Results saved to 'optimized_trajectory.g2o'." << std::endl;

    //return the optimized graph as a string
    std::ifstream file_opt("optimized_trajectory.g2o");
    std::string optimized_graph((std::istreambuf_iterator<char>(file_opt)), std::istreambuf_iterator<char>());

    //Close open streams
    file_opt.close();
    file_cov.close();

    return optimized_graph;
}



