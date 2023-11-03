import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import dok_array

from utilities import Plotter
from solver import UFLExactSolve, PrimalDualSolver


class Customer():
    def __init__(self, c_id, ordered_facilities):
        self.id = c_id
        self.alpha = 0.0

        self.affiliates = []
        self.ordered_facilities = ordered_facilities # list of non-affiliate facilities in ascending order of distance 

        self.witness = None
        self.is_connected = False

    def get_next_facility(self, pop=False):
        if len(self.ordered_facilities)>0:
            idx = self.ordered_facilities[0]
            if pop:
                self.ordered_facilities = self.ordered_facilities[1:]
        else:
            idx = None
        return idx


class Facility():
    def __init__(self, f_id, opening_cost):
        self.id = f_id
        self.opening_cost = opening_cost

        self.open = False
        self.affiliates = []


class World():
    def __init__(self, num_f, num_c, facility_loc, customer_loc, transport_costs, opening_costs):
        assert( transport_costs.shape[0]== num_f)
        assert( transport_costs.shape[1]== num_c)
        assert( opening_costs.shape[0]== num_f)

        self.num_f = num_f
        self.num_c = num_c
        self.facility_loc = facility_loc
        self.customer_loc = customer_loc
        self.transport_costs = transport_costs
        self.opening_costs = opening_costs

        self.betas = dok_array((num_f, num_c), dtype=np.float32)

        # create facilities #
        self.facilities = []
        for i in range(num_f):
            self.facilities.append( Facility(i, opening_costs[i]) )

        # create customers #
        self.customers = []
        for j in range(num_c):
            pov_ordered_facilities = np.argsort(transport_costs[:,j])
            self.customers.append( Customer(j, pov_ordered_facilities) )
            



class WorldGenerator():
    def __init__(self, rng):
        self.rng = rng

    def generate(self, m, n, max_opening_cost):
        customer_loc = rng.random_sample((2,n))
        facility_loc = rng.random_sample((2,m))

        # calculate transportation costs #
        C = np.repeat(customer_loc[:, np.newaxis, :], m, axis=1)
        F = np.repeat(facility_loc[:, :, np.newaxis], n, axis=2)
        transport_costs = np.sum( (C-F)**2, axis=0 )
        transport_costs = np.sqrt(transport_costs) # [num_f x num_c]

        # generate opening costs for facilities #
        opening_costs = max_opening_cost * rng.random_sample(m)

        return World(m, n, facility_loc, customer_loc, transport_costs, opening_costs)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed (default 0).',
        type=int,
        default=72,
    )
    parser.add_argument(
        '-f', '--num_f',
        help='Number of facilities (default 2).',
        type=int,
        default=2,
    )
    parser.add_argument(
        '-c', '--num_c',
        help='Number of cutomers (default 4).',
        type=int,
        default=4,
    )
    parser.add_argument(
        '-m', '--max_cost',
        help='Maximum opening cost (default 0.5).',
        type=float,
        default=0.5,
    )
    args = parser.parse_args()

    # random number generator #
    rng = np.random.RandomState(seed=args.seed)

    # generate data #
    generator = WorldGenerator(rng)
    world = generator.generate(args.num_f, args.num_c, max_opening_cost=args.max_cost)

    # exact solver #
    solver = UFLExactSolve()
    solver.create_problem(world)
    solver.solve()
    optimal_solution = solver.get_solution()

    # primal-dual solver #
    solver = PrimalDualSolver()
    solver.create_problem(world)
    solver.phase_1()
    solver.phase_2()
    primaldual_solution = solver.get_solution()

    # plot solutions #
    plotter = Plotter(world)
    plotter.plot_3_solutions(optimal_solution, primaldual_solution)
