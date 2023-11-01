import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

class Solution():
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Event():
    def __init__(self, nxt, time2event):
        self.nxt = nxt
        self.time2event = time2event



class PrimalDualSolver():
    def __init__(self):
        self._world = None

    def create_problem(self, world):
        self._world = world

    def next_tight_edge_event(self):
        nxt = None
        min_slack = np.inf
        for c in self._world.customers:
            if not c.is_connected:
                c_id = c.id
                f_id = c.get_next_facility()
                if f_id is not None:
                    slack = self._world.transport_costs[f_id][c_id] - c.alpha
                    if slack < min_slack:
                        min_slack = slack
                        nxt = c_id
        
        return Event(nxt, time2event=min_slack)


    def next_opening_event(self):
        nxt = None
        min_slack = np.inf
        for f in self._world.facilities:
            if not f.open:
                f_id = f.id
                if f.affiliates:
                    paid = np.sum([self._world.betas[f_id, c_id]  for c_id in f.affiliates])
                    slack = (f.opening_cost - paid) / len(f.affiliates)
                else:
                    slack = np.inf
                if slack < min_slack:
                    min_slack = slack
                    nxt = f_id

        return Event(nxt, time2event=min_slack)
                    

    def forward(self, timestep):
        for c in self._world.customers:
            if not c.is_connected:
                c_id = c.id
                c.alpha += timestep 
                for f_id in c.affiliates:
                    self._world.betas[f_id, c_id] += timestep

    def phase_1(self):
        unconnected_customers = [j for j in range(self._world.num_c)]

        while unconnected_customers:

            event_tight = self.next_tight_edge_event()
            event_open = self.next_opening_event()

            if event_tight.time2event < event_open.time2event:
                self.forward( event_tight.time2event )
                c_id = event_tight.nxt
                f_id = self._world.customers[c_id].get_next_facility(pop=True)
                if self._world.facilities[f_id].open:
                    self._world.customers[c_id].is_connected = True
                    self._world.customers[c_id].witness = f_id
                    unconnected_customers.remove( c_id )
                else:
                    self._world.customers[c_id].affiliates.append(f_id)
                    self._world.facilities[f_id].affiliates.append(c_id)
            else:
                self.forward( event_open.time2event )
                f_id = event_open.nxt
                self._world.facilities[f_id].open = True
                for c_id in self._world.facilities[f_id].affiliates:
                    self._world.customers[c_id].is_connected = True
                    self._world.customers[c_id].witness = f_id
                    unconnected_customers.remove( c_id )



    def phase_2(self):

        open_facilities = [i for i in range(self._world.num_f) if self._world.facilities[i].open]

        model = Model("Independent Set")
        model.setParam('display/verblevel', 0)

        # create variables #
        z = {}
        for i in open_facilities:
            z[i] = model.addVar(vtype="B", name=f"z({i})")

        # create constraints #
        for c in self._world.customers:
            open_affiliates = [i for i in c.affiliates if i in open_facilities]
            edges = list(combinations(open_affiliates, 2))
            for u, v in edges:
                model.addCons(z[u] + z[v] <= 1, f"({u}-{v})")

        # set objective value #
        model.setObjective(quicksum(z[i] for i in open_facilities), "minimize")

        # solve #
        model.optimize()



class UFLExactSolve():
    def __init__(self):
        self._model = None
        self._xvars = None
        self._yvars = None
        self._solution = None
        self._model_size = (None, None)

    def create_problem(self, world):
        self._solution = None

        num_f = world.num_f
        num_c = world.num_c
        C = world.transport_costs
        F = world.opening_costs

        model = Model("UFL")
        model.setParam('display/verblevel', 0)

        # create variables #
        x, y = {},{}
        for i in range(num_f):
            y[i] = model.addVar(vtype="B", name=f"y({i})")
            for j in range(num_c):
                x[i,j] = model.addVar(vtype="B", name=f"x({i},{j})")
            

        # service constraints #
        for j in range(num_c):
            model.addCons(quicksum(x[i,j] for i in range(num_f)) >= 1, f"customer_({j})")

        # opening constraints #
        for i in range(num_f):
            for j in range(num_c):
                model.addCons(x[i,j] <= y[i], f"c_({j})_f_({i})")

        # set objective value #
        model.setObjective(quicksum(C[i][j]*x[i,j] for i in range(num_f) for j in range(num_c))
                      + quicksum(F[i]*y[i] for i in range(num_f)), "minimize")

        self._model = model
        self._xvars = x 
        self._yvars = y
        self._model_size = (num_f, num_c)

    def solve(self):
        if self._model is not None:
            self._model.optimize()
            
            num_f, num_c = self._model_size
            
            ysolution = {i: None for i in range(num_f)}
            for i in range(num_f):
                ysolution[i] = self._model.getVal(self._yvars[i])

            xsolution = {j: None for j in range(num_c)}
            for j in range(num_c):
                for i in range(num_f):
                    if self._model.getVal(self._xvars[i,j]):
                        assert(xsolution[j] is None)
                        xsolution[j] = i
                assert(xsolution[j] is not None)

            self._solution = Solution(xsolution, ysolution)

    def get_solution(self):
        return self._solution
