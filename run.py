import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import dok_array

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



class Plotter():
    def __init__(self, world):
        self._world = world
        self.customer_loc = world.customer_loc
        self.facility_loc = world.facility_loc

    def plot_problem(self):
        fig, ax = plt.subplots()
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.scatter(self.customer_loc[0], self.customer_loc[1], color='peru')
        ax.scatter(self.facility_loc[0], self.facility_loc[1], color='slateblue')
        plt.show()

    def plot_solution(self, solution, title=None):
        fig, ax = plt.subplots()
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        if title is not None:
            ax.set_title(title)
        
        # plot all facility locations #
        ax.scatter(self.facility_loc[0], self.facility_loc[1], alpha=0.4, color='slateblue')

        # plot used edges #
        for j in range(self._world.num_c):
            i = solution.x[j]
            xx = [self.customer_loc[0][j], self.facility_loc[0][i]]
            yy = [self.customer_loc[1][j], self.facility_loc[1][i]]
            ax.plot(xx, yy, color='k')

        # plot open facilities and customers #
        ax.scatter(self.customer_loc[0], self.customer_loc[1], color='peru')
        for i in range(self._world.num_f):
            if solution.y[i]:
                ax.scatter(self.facility_loc[0][i], self.facility_loc[1][i], alpha=1.0, color='slateblue')

        plt.show()

    def plot_phase1(self):
        fig, ax = plt.subplots()
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_title('Phase 1 solution')

        # plot all facility locations #
        ax.scatter(self.facility_loc[0], self.facility_loc[1], alpha=0.4, color='slateblue')

        # plot special edges #
        for c in self._world.customers:
            c_id = c.id
            for f_id in c.affiliates: 
                xx = [self.customer_loc[0][c_id], self.facility_loc[0][f_id]]
                yy = [self.customer_loc[1][c_id], self.facility_loc[1][f_id]]
                ax.plot(xx, yy, color='k', linestyle='--')

        # plot used edges #
        for c in self._world.customers:
            c_id = c.id
            f_id = c.witness 
            xx = [self.customer_loc[0][c_id], self.facility_loc[0][f_id]]
            yy = [self.customer_loc[1][c_id], self.facility_loc[1][f_id]]
            ax.plot(xx, yy, color='k')

        # plot open facilities and customers #
        ax.scatter(self.customer_loc[0], self.customer_loc[1], color='peru')
        for f in self._world.facilities:
            if f.open:
                ax.scatter(self.facility_loc[0][f.id], self.facility_loc[1][f.id], alpha=1.0, color='slateblue')

        plt.show()




if __name__ == "__main__":
    rng = np.random.RandomState(seed=72)

    # generate data #
    generator = WorldGenerator(rng)
    world = generator.generate(m=2, n=4, max_opening_cost=0.4)

    # exact solver #
    solver = UFLExactSolve()
    solver.create_problem(world)
    solver.solve()
    optimal_solution = solver.get_solution()

    # primal-dual solver #
    solver = PrimalDualSolver()
    solver.create_problem(world)
    solver.phase_1()

    # for f in world.facilities:
    #     print(f"Facility {f.id}: {f.open}")
    # print(" ")
    # for c in world.customers:
    #     print(f"Customer {c.id} is connected to {c.witness}")
    #     print(f"also paying for {c.affiliates}")


    # plotter = Plotter(world)
    # plotter.plot_solution(optimal_solution, title='Optimal solution')
    # plotter.plot_phase1()