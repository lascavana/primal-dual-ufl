import matplotlib.pyplot as plt


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

    def plot_solution(self, solution, ax=None, title=None):

        if ax is None:
            fig, ax = plt.subplots()
            plot_now = True
        else:
            plot_now = False
        
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

        if plot_now:
            plt.show()

    def plot_phase1(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots()
            plot_now = True
        else:
            plot_now = False

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

        if plot_now:
            plt.show()

    def plot_3_solutions(self, opt_sol, pd_sol):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

        self.plot_solution(opt_sol, ax=ax1, title='Optimal solution')
        self.plot_phase1(ax=ax2)
        self.plot_solution(pd_sol, ax=ax3, title='Primal-dual solution')

        plt.show()



