import numpy as np 


class Customer():
    def __init__(self, c_id, ordered_facilities):
        self.id = c_id
        self.alpha = 0.0
        
        self.witness = None
        self.is_connected = False

        self.affiliates = []
        self.ordered_facilities = ordered_facilities # list of non-affiliate facilities in ascending order of distance

    def get_next_facility(self, pop=False):
        idx = self.ordered_facilities[0]
        if pop:
            self.ordered_facilities.pop(0)
        return idx


class Facility():
    def __init__(self, f_id, opening_cost):
        self.id = f_id
        self.opening_cost = opening_cost

        self.open = False
        self.affiliates = []
        

class Event():
    def __init__(self, nxt, time2event):
        self.nxt = nxt
        self.time2event = time2event



class World():
    def __init__(self, num_f, num_c, transport_costs, opening_costs):
        assert( transport_costs.shape[0]== num_f)
        assert( transport_costs.shape[1]== num_c)
        assert( opening_costs.shape[0]== num_f)

        self.num_f = num_f
        self.num_c = num_c
        self.transport_costs = transport_costs

        self.betas = scipy.dok_array((num_f, num_c), dtype=np.float32)

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
        transport_costs = np.sqrt(transport_costs)

        # generate opening costs for facilities #
        opening_costs = rng.randint(0, max_opening_cost, size=m)

        return World(m, n, transport_costs, opening_costs)




class Solver():
    def __init__(self):
        self._world = None

    def load_world(self, world):
        self._world = world

    def next_tight_edge_event(self):
        nxt = None
        min_slack = np.inf
        for c in self._world.customers:
            if not c.connected:
                c_id = c.id
                f_id = c.get_next_facility()
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
                else:
                    paid = 0.0
                slack = f.opening_cost - paid
                if to_pay < min_slack:
                    min_slack = slack
                    nxt = f_id

        return Event(nxt, time2event=min_slack)
                    

    def forward(self, timestep):
        for c in self.world.customers:
            if not c.connected:
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
                forward( event_tight.time2event )
                c_id = event_tight.nxt
                f_id = self._world.customers[c_id].get_next_facility(pop=True)
                self._world.customers[c_id].affiliates.append(f_id)
                self._world.facilities[f_id].affiliates.append(c_id)
            else:
                forward( event_open.time2event )
                f_id = event_open.nxt
                self._world.facilities[f_id].open = True
                for c_id in self._world.facilities[f_id].affiliates:
                    self._world.customers[c_id].connected = True
                    self._world.customers[c_id].witness = f_id
                    self.unconnected_customers.remove( c_id )

    def phase_2(self):
        pass




rng = np.random.RandomState(seed=72)
gen = WorldGenerator(rng)
gen.generate(m=2, n=3)