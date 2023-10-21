import mesa
import random
import numpy as np
import seaborn as sns
from globals import Gl

class Car(mesa.Agent):
  def __init__(self, unique_id, model):
    # Pass the parameters to the parent class.
    super().__init__(unique_id, model)
    self.in_pos = {"": None}  # {name of road/junction: cell position}
    self.travel_time = 0
    self.history = [] # gets appended by {route: travel_time} after destination
    self.selected_route = None
    self.expected_travel_R14 = 0
    self.expected_travel_R23 = 0
    self.expected_travel_R153 = 0
    self.waited_at_junction = 0
    self.burn_in_phase = True  # during burn-in, try each route at least once

  def move(self) -> None:
    [(name, pos)] = self.in_pos.items()
    if not name:
      raise Exception("Can't move car because it's not in any infrastructure")
    self.model.advance_agent(self)

  def update_data(self) -> None:
    self.update_travel_time()
    self.update_t_info()
    self.strat_select_route()

  def update_travel_time(self) -> None:
    if self.selected_route == None:
      # very first run, don't count it due to how agents are spawned
      return
    if self.get_structure_name() is Gl.E0:
      if self.travel_time == 0:
        # waiting in E0, do nothing
        return
      self.history.append([self.selected_route, self.travel_time])
      self.sync_travel_time_to_model()
      if len(self.history) > 30:
        # forget oldest time
        del self.history[0]
      self.travel_time = 0
    else:
      self.travel_time += 1

  def sync_travel_time_to_model(self) -> None:
    if self.selected_route == None:
      return
    if self.selected_route == Gl.R14:
      self.model.most_recent_travel_time_R14 = self.travel_time
      self.model.counter_traveled_R14 += 1
    elif self.selected_route == Gl.R23:
      self.model.most_recent_travel_time_R23 = self.travel_time
      self.model.counter_traveled_R23 += 1
    else:
      assert(self.selected_route == Gl.R153)
      self.model.most_recent_travel_time_R153 = self.travel_time
      self.model.counter_traveled_R153 += 1


  def update_t_info(self) -> None:
    if not self.get_structure_name() is Gl.E0:
      return
    counter_R14 = 0
    total_travel_time_R14 = 0
    counter_R23 = 0
    total_travel_time_R23 = 0
    counter_R153 = 0
    total_travel_time_R153 = 0
    for hist in self.history:
      route = hist[0]
      travel_time = hist[1]
      if route is Gl.R14:
        counter_R14 += 1
        total_travel_time_R14 += travel_time
      elif route is Gl.R23:
        counter_R23 += 1
        total_travel_time_R23 += travel_time
      elif route is Gl.R153:
        counter_R153 += 1
        total_travel_time_R153 += travel_time

    if counter_R14 != 0:
      self.expected_travel_R14 = total_travel_time_R14 / counter_R14
    if counter_R23 != 0:
      self.expected_travel_R23 = total_travel_time_R23 / counter_R23
    if counter_R153 != 0:
      self.expected_travel_R153 = total_travel_time_R153 / counter_R153


  def strat_select_route(self) -> None:
    """When an agent has finished its route and is on edge E0, it will select a
    route for its next round. 2 cases:
    1) burn-in phase, take each route at least once
    2) select ideal route based on memory,
    """
    if not self.get_structure_name() is Gl.E0:
      return
    # 1)
    if self.burn_in_phase:
      self.selected_route = random.choice(self.model.possible_routes())
      if len(self.history) == 30:
        self.burn_in_phase = False
    # 2)
    elif random.random() > Gl.PROB_INFO:
      self.selected_route = random.choice(self.model.possible_routes())
    elif self.expected_travel_R153 == 0:
      # 4link
      if Gl.DELTA_T < abs(self.expected_travel_R14 - self.expected_travel_R23):
        self.selected_route = self.model.possible_routes()[np.argmin([self.expected_travel_R14, self.expected_travel_R23])]
    elif (Gl.DELTA_T >= abs(self.expected_travel_R14 - self.expected_travel_R23)
        + abs(self.expected_travel_R14 - self.expected_travel_R153)
        + abs(self.expected_travel_R23 - self.expected_travel_R153)):
      # don't switch route, 5link
      return
    else:
      # switch route, 5link
      route_index = np.argmin([self.expected_travel_R14, self.expected_travel_R23,
                               self.expected_travel_R153])
      self.selected_route = self.model.possible_routes()[route_index]


  def strat_maybe_change_route(self, is4link) -> None:
    """When an agent is at a junction and the next spot on the agent's desired
    route is occupied, there's a chance that the agent will change its route"""
    assert(self.get_structure_name() is Gl.J1 or self.get_structure_name() is Gl.J2)
    self.waited_at_junction += 1
    # 4link
    if self.get_structure_name() is Gl.J1 and is4link:
      self.strat_maybe_change_route_4link()
    # 5link
    # see "Bittihn, 2018" apendix pseudocode
    elif self.get_structure_name() is Gl.J2:
      self.strat_maybe_change_route_J2()
    else:
      self.strat_maybe_change_route_J1()


  def strat_maybe_change_route_4link(self) -> None:
    if self.selected_route == Gl.R14:
      if self.expected_travel_R14 < self.expected_travel_R23:
        if self.waited_at_junction > (self.expected_travel_R23 - self.expected_travel_R14) * Gl.J2_WAIT_THRESHOLD:
          self.selected_route = Gl.R23
          self.waited_at_junction = 0
      else:
        self.selected_route = Gl.R23
        self.waited_at_junction = 0
    else:
      assert(self.selected_route == Gl.R23)
      if self.expected_travel_R23 < self.expected_travel_R14:
        if self.waited_at_junction > (self.expected_travel_R14 - self.expected_travel_R23) * Gl.J2_WAIT_THRESHOLD:
          self.selected_route = Gl.R14
          self.waited_at_junction = 0
      else:
        self.selected_route = Gl.R14
        self.waited_at_junction = 0


  def strat_maybe_change_route_J1(self) -> None: # 2 functions depending on which junction the agent is
    if self.selected_route == Gl.R14 or self.selected_route == Gl.R153:
      if self.selected_route == Gl.R14:
        if self.expected_travel_R14 < self.expected_travel_R23:
          if self.waited_at_junction > (self.expected_travel_R23 - self.expected_travel_R14) * Gl.J1_WAIT_THRESHOLD:
            self.selected_route = Gl.R23
            self.waited_at_junction = 0
        else:
          self.selected_route = Gl.R23
          self.waited_at_junction = 0
      else:
        assert(self.selected_route == Gl.R153)
        if self.expected_travel_R153 < self.expected_travel_R23:
          if self.waited_at_junction > (self.expected_travel_R23 - self.expected_travel_R153) * Gl.J1_WAIT_THRESHOLD:
            self.selected_route = Gl.R23
            self.waited_at_junction = 0
        else:
          self.selected_route = Gl.R23
          self.waited_at_junction = 0
    else:
      assert(self.selected_route == Gl.R23)
      if (self.expected_travel_R23 < self.expected_travel_R14) and (self.expected_travel_R23 < self.expected_travel_R153):
        if self.expected_travel_R14 < self.expected_travel_R153:
          if self.waited_at_junction > (self.expected_travel_R14 - self.expected_travel_R23) * Gl.J1_WAIT_THRESHOLD:
            self.selected_route = Gl.R14
            self.waited_at_junction = 0
        else:
          assert(self.expected_travel_R153 <= self.expected_travel_R14)
          if self.waited_at_junction > (self.expected_travel_R153 - self.expected_travel_R23) * Gl.J1_WAIT_THRESHOLD:
            self.selected_route = Gl.R153
            self.waited_at_junction = 0
      else:
        if self.expected_travel_R14 < self.expected_travel_R153:
          self.selected_route = Gl.R14
          self.waited_at_junction = 0
        elif self.expected_travel_R153 < self.expected_travel_R14:
          self.selected_route = Gl.R153
          self.waited_at_junction = 0
        else:
          self.selected_route = random.choice([Gl.R14, Gl.R153])
          self.waited_at_junction = 0


  def strat_maybe_change_route_J2(self) -> None:
    if self.selected_route == Gl.R14:
      if self.expected_travel_R14 < self.expected_travel_R153:
        if self.waited_at_junction > (self.expected_travel_R153 - self.expected_travel_R14) * Gl.J2_WAIT_THRESHOLD:
          self.selected_route = Gl.R153
          self.waited_at_junction = 0
      else:
        self.selected_route = Gl.R153
        self.waited_at_junction = 0
    else:
      assert(self.selected_route == Gl.R153)
      if self.expected_travel_R153 < self.expected_travel_R14:
        if self.waited_at_junction > (self.expected_travel_R14 - self.expected_travel_R153) * Gl.J2_WAIT_THRESHOLD:
          self.selected_route = Gl.R14
          self.waited_at_junction = 0
      else:
        self.selected_route = Gl.R14
        self.waited_at_junction = 0

  def get_structure_name(self):
    return next(iter(self.in_pos))

  def print_history(self) -> None:
    print(f"I'm car {self.unique_id}, my travel history: {self.history}")

  def step(self) -> None:
    self.move()
    self.update_data()
    if self.model.verbose:
      print(f"I'm car {self.unique_id} in {self.in_pos} at {self.pos}\n")
#%%
class Tasep():
  """User defined class for TASEPs/edges/roads"""

  def __init__(self, L, name, pos_start, pos_end, verbose=False):
    assert L > 0
    self.length = L
    self.name = name
    self.pos_start = pos_start
    self.verbose = verbose
    self.heading = np.array(pos_end) - np.array(self.pos_start)
    self.cells = [self.default_val()] * self.length

  @staticmethod
  def default_val() -> None:
    """Default value for new cell elements."""
    return None

  @staticmethod
  def is_junction() -> bool:
    return False

  def is_cell_empty(self, id: int = 0) -> bool:
    """Returns a bool of the contents of a cell."""
    assert id >= 0 and id < self.length
    return self.cells[id] == self.default_val()

  def is_last_id(self, id: int) -> bool:
    assert id >= 0 and id < self.length
    return id == self.length - 1

  def _add_agent(self, agent: mesa.Agent, id: int) -> None:
    """Add agent at a cell id, for internal use"""
    if self.is_cell_empty(id):
      self.cells[id] = agent
      self.set_agent_pos(agent, id)
    else:
      raise Exception("Cell not empty")

  def add_agent(self, agent: mesa.Agent) -> None:
    """Add agent at the start if able"""
    self._add_agent(agent, 0)

  def advance_agent(self, agent: mesa.Agent) -> None:
    """advanced agent by 1 if able"""
    [(name, id)] = agent.in_pos.items()
    assert name is self.name

    if id == self.length - 1:
      #TODO add a way such that a car/agent goes from last place in a cell to connected junction
      raise Exception("Can't advance, agent at the last cell")
    else:
      if self.is_cell_empty(id + 1):
        self.cells[id + 1] = self.cells[id]
        self.cells[id] = self.default_val()
        self.set_agent_pos(agent, id + 1)
      elif self.verbose:
        print(f"Can't advance agent {agent.unique_id} because cell {id + 1} is not empty.")

  def set_agent_pos(self, agent: mesa.Agent, id: int):
    agent.pos = self.pos_start + id / self.length * self.heading
    agent.in_pos = {self.name: id}

  def remove_last_id(self) -> None:
    self.cells[self.length - 1] = self.default_val()
    # note, doesn't reset agent position

#%%
class Junction():
  """"""

  def __init__(self, name, pos):
    self.name = name
    self.pos = pos
    self.cell = None

  @staticmethod
  def is_junction() -> bool:
    return True

  def is_cell_empty(self) -> bool:
    """Returns a bool of the content of the cell."""
    return self.cell == None

  def add_agent(self, agent: mesa.Agent) -> None:
    """Add agent if able"""
    if self.is_cell_empty():
      self.cell = agent
      agent.pos = self.pos
      agent.in_pos = {self.name: 0}
    else:
      raise Exception("Cell not empty")

  def remove_last_id(self) -> None:
    self.cell = None
    # note, doesn't reset agent position

#%%
class Model_4link(mesa.Model):
  """"""

  def __init__(self, N, verbose=False):
    self.num_agents = N
    self.verbose = verbose
    self.ticks = 0
    # Create scheduler and assign it to the model
    self.schedule = mesa.time.RandomActivation(self)
    self.infrastructure = {
        Gl.J1: Junction(Gl.J1, pos=(50, 0)),
        Gl.J2: Junction(Gl.J2, pos=(0, 100)),
        Gl.J3: Junction(Gl.J3, pos=(100, 100)),
        Gl.J4: Junction(Gl.J4, pos=(50, 200)),
        Gl.E1: Tasep(100, Gl.E1, pos_start=(50, 0), pos_end=(0, 100), verbose=self.verbose),
        Gl.E2: Tasep(500, Gl.E2, pos_start=(50, 0), pos_end=(100, 100), verbose=self.verbose),
        Gl.E3: Tasep(100, Gl.E3, pos_start=(100, 100), pos_end=(50, 200), verbose=self.verbose),
        Gl.E4: Tasep(500, Gl.E4, pos_start=(0, 100), pos_end=(50, 200), verbose=self.verbose),
        # no L5 yet
        Gl.E0: Tasep(1, Gl.E0, pos_start=(120, 100), pos_end=(120, 100)),
        Gl.E_SPAWN: Tasep(self.num_agents, Gl.E_SPAWN, pos_start=(0, 0), pos_end=(0, 0), verbose=self.verbose),
    }
    self.infra_connections_R14 = {
        Gl.J1: Gl.E1,
        Gl.J2: Gl.E4,
        Gl.J4: Gl.E0,
        Gl.E1: Gl.J2,
        Gl.E4: Gl.J4,
        Gl.E0: Gl.J1,
    }
    self.infra_connections_R23 = {
        Gl.J1: Gl.E2,
        Gl.J3: Gl.E3,
        Gl.J4: Gl.E0,
        Gl.E2: Gl.J3,
        Gl.E3: Gl.J4,
        Gl.E0: Gl.J1,
    }
    self.most_recent_travel_time_R14 = None
    self.counter_traveled_R14 = 0
    self.most_recent_travel_time_R23 = None
    self.counter_traveled_R23 = 0
    self.datacollector = mesa.DataCollector(model_reporters={
        "route_R14": "most_recent_travel_time_R14",
        "route_R23": "most_recent_travel_time_R23",
        "counter_R14": "counter_traveled_R14",
        "counter_R23": "counter_traveled_R23",
    })


  @staticmethod
  def possible_routes():
    return Gl.ROUTES_4LINK


  def initialize_agents_positions(self) -> None:
    """Create agents and initialize them to the schedule and infrastructure"""
    for idx in range(self.num_agents):
      car = Car(idx, self)
      # Add the agent to the scheduler
      self.schedule.add(car)
      self.infrastructure[Gl.E_SPAWN]._add_agent(car, idx)


  def advance_agent(self, agent: mesa.Agent) -> None:
    [(infra_name, id)] = agent.in_pos.items()
    structure = self.infrastructure[infra_name]
    if structure.is_junction() or structure.is_last_id(id):
      self.advance_agent_infrastructure(agent)
    else:
      structure.advance_agent(agent)

  def advance_agent_infrastructure(self, agent: mesa.Agent) -> None:
    current_infra_name = agent.get_structure_name()
    next_infra_name = None
    match agent.selected_route:
      case Gl.R14:
        next_infra_name = self.infra_connections_R14[current_infra_name]
      case Gl.R23:
        next_infra_name = self.infra_connections_R23[current_infra_name]
      case None:
        assert(agent.travel_time == 0)
        next_infra_name = Gl.E0 # selected_route will be set in E0
      case _:
        raise Exception(f"Unknown route {agent.selected_route}")

    if self.verbose:
      print(f"try to move from {current_infra_name} to {next_infra_name}")
    elif self.infrastructure[next_infra_name].is_cell_empty():
      self.infrastructure[current_infra_name].remove_last_id()
      self.infrastructure[next_infra_name].add_agent(agent)
    else:
      if self.verbose:
        print(f"next infrastructure {next_infra_name} not empty")
      if current_infra_name is Gl.J1 or current_infra_name is Gl.J2:
        agent.strat_maybe_change_route(is4link=True)

  def collect(self) -> None:
    if self.most_recent_travel_time_R14 is not None and self.most_recent_travel_time_R23 is not None:
      self.datacollector.collect(self)

  def step(self) -> None:
    """Advance the model by one step."""
    # The model's step will go here for now this will call the step method of each agent
    self.schedule.step()
    self.collect()
    if self.verbose:
      print("----------")
#%%
class Model_5link(Model_4link):
  """"""
  def __init__(self, N, verbose=False):
    super().__init__(N, verbose)
    self.infrastructure[Gl.E5] = Tasep(37, Gl.E5, pos_start=(0, 100), pos_end=(100, 100), verbose=self.verbose)
    self.infra_connections_R153 = {
        Gl.J1: Gl.E1,
        Gl.J2: Gl.E5,
        Gl.J3: Gl.E3,
        Gl.J4: Gl.E0,
        Gl.E1: Gl.J2,
        Gl.E3: Gl.J4,
        Gl.E5: Gl.J3,
        Gl.E0: Gl.J1,
    }
    self.most_recent_travel_time_R153 = None
    self.counter_traveled_R153 = 0
    self.datacollector = mesa.DataCollector(model_reporters={
        "route_R14": "most_recent_travel_time_R14",
        "route_R23": "most_recent_travel_time_R23",
        "route_R153": "most_recent_travel_time_R153",
        "counter_R14": "counter_traveled_R14",
        "counter_R23": "counter_traveled_R23",
        "counter_R153": "counter_traveled_R153",
    })


  @staticmethod
  def possible_routes():
    return Gl.ROUTES_5LINK

  # override
  def advance_agent_infrastructure(self, agent: mesa.Agent) -> None:
    current_infra_name = agent.get_structure_name()
    next_infra_name = None
    match agent.selected_route:
      case Gl.R14:
        next_infra_name = self.infra_connections_R14[current_infra_name]
      case Gl.R23:
        next_infra_name = self.infra_connections_R23[current_infra_name]
      case Gl.R153:
        next_infra_name = self.infra_connections_R153[current_infra_name]
      case None:
        assert(agent.travel_time == 0)
        next_infra_name = Gl.E0 # selected_route will be set in E0
      case _:
        raise Exception(f"Unknown route {agent.selected_route}")

    if self.verbose:
      print(f"try to move from {current_infra_name} to {next_infra_name}")
    if self.infrastructure[next_infra_name].is_cell_empty():
      self.infrastructure[current_infra_name].remove_last_id()
      self.infrastructure[next_infra_name].add_agent(agent)
    else:
      if self.verbose:
        print(f"next infrastructure {next_infra_name} not empty")
      if current_infra_name is Gl.J1 or current_infra_name is Gl.J2:
        agent.strat_maybe_change_route(is4link=False)

  # override
  def collect(self) -> None:
    if (self.most_recent_travel_time_R14 is not None and
        self.most_recent_travel_time_R23 is not None and
        self.most_recent_travel_time_R153 is not None):
      self.datacollector.collect(self)

#%%
class Bus(mesa.Agent):
  def __init__(self, unique_id, model):
    # Pass the parameters to the parent class.
    super().__init__(unique_id, model)
    self.in_pos = {"": None}  # {name of road/junction: cell position}
    self.travel_time = 0
    self.selected_route = Gl.R153

  def move(self) -> None:
    [(name, pos)] = self.in_pos.items()
    if not name:
      raise Exception("Can't move car because it's not in any infrastructure")
    self.model.advance_agent(self) # does this work?

  def update_data(self) -> None:
    self.update_travel_time()

  def update_travel_time(self) -> None:
    if self.get_structure_name() is Gl.E0:
      if self.travel_time == 0:
        # waiting in E0, do nothing
        return
      self.sync_travel_time_to_model()
      self.travel_time = 0
    else:
      self.travel_time += 1

  def sync_travel_time_to_model(self) -> None:
    assert(self.selected_route == Gl.R153)
    self.model.most_recent_travel_time_R153 = self.travel_time
    self.model.counter_traveled_R153 += 1

  def get_structure_name(self):
    return next(iter(self.in_pos))

  def print_history(self) -> None:
    print(f"I'm bus {self.unique_id}, I have no history")

  def strat_maybe_change_route(self, is4link) -> None:
    # busses do not change route
    pass

  def step(self) -> None:
    self.move()
    self.update_data()
    if self.model.verbose:
      print(f"I'm bus {self.unique_id} in {self.in_pos} at {self.pos}\n")
#%%
class Model_5link_with_bus(Model_5link):
  """"""
  def __init__(self, n_car, n_bus, verbose=False):
    super().__init__(n_car, verbose)
    self.n_car = n_car  # number of cars
    self.n_bus = n_bus  # number of busses
    # add spawn edge for busses
    self.infrastructure[Gl.E_SPAWN_BUS] = Tasep(self.n_bus, Gl.E_SPAWN_BUS, pos_start=(0, 0), pos_end=(0, 0), verbose=self.verbose)
    self.infra_connections_R153[Gl.E_SPAWN_BUS] = Gl.E0

  @staticmethod
  def possible_routes(isBus=False):
    if isBus:
      return [Gl.R153]
    #else
    return Gl.ROUTES_4LINK

  #overide
  def initialize_agents_positions(self) -> None:
    """Create car and bus agents and initialize them to the schedule and infrastructure"""
    super().initialize_agents_positions()
    for idx in range(self.n_bus):
      bus = Bus(self.n_car + idx, self)
      self.schedule.add(bus)
      self.infrastructure[Gl.E_SPAWN_BUS]._add_agent(bus, idx)

  # unfortunate to have to copy this all, could maybe improve? (is4link=True)
  # override
  def advance_agent_infrastructure(self, agent: mesa.Agent) -> None:
    current_infra_name = agent.get_structure_name()
    next_infra_name = None
    match agent.selected_route:
      case Gl.R14:
        next_infra_name = self.infra_connections_R14[current_infra_name]
      case Gl.R23:
        next_infra_name = self.infra_connections_R23[current_infra_name]
      case Gl.R153:
        next_infra_name = self.infra_connections_R153[current_infra_name]
      case None:
        assert(agent.travel_time == 0)
        next_infra_name = Gl.E0 # selected_route will be set in E0
      case _:
        raise Exception(f"Unknown route {agent.selected_route}")

    if self.verbose:
      print(f"try to move from {current_infra_name} to {next_infra_name}")
    if self.infrastructure[next_infra_name].is_cell_empty():
      self.infrastructure[current_infra_name].remove_last_id()
      self.infrastructure[next_infra_name].add_agent(agent)
    else:
      if self.verbose:
        print(f"next infrastructure {next_infra_name} not empty")
      if current_infra_name is Gl.J1 or current_infra_name is Gl.J2:
        agent.strat_maybe_change_route(is4link=True) # cars may only use 4link
        # busses don't switch at all
#%%
class Car_toll(Car):
  def __init__(self, unique_id, model):
    # Pass the parameters to the parent class.
    super().__init__(unique_id, model)
    self.cost_sensitivity = 1
    self.toll_cost = self.model.base_toll_cost * self.cost_sensitivity

  #override
  def update_travel_time(self) -> None:
    if self.selected_route == None:
      # very first run, don't count it due to how agents are spawned
      return
    if self.get_structure_name() is Gl.E0:
      if self.travel_time == 0:
        # waiting in E0, do nothing
        return
      if self.selected_route == Gl.R153:
        self.history.append([self.selected_route, self.travel_time + self.toll_cost])
      else:
        self.history.append([self.selected_route, self.travel_time])
      self.sync_travel_time_to_model()
      if len(self.history) > 30:
        # forget oldest time
        del self.history[0]
      self.travel_time = 0
    else:
      self.travel_time += 1

  #override
  def sync_travel_time_to_model(self) -> None:
    if self.selected_route == None:
      return
    if self.selected_route == Gl.R14:
      self.model.most_recent_travel_time_R14 = self.travel_time
      self.model.counter_traveled_R14 += 1
    elif self.selected_route == Gl.R23:
      self.model.most_recent_travel_time_R23 = self.travel_time
      self.model.counter_traveled_R23 += 1
    else:
      assert(self.selected_route == Gl.R153)
      self.model.most_recent_travel_time_R153 = self.travel_time
      self.model.experienced_travel_time_R153 = self.travel_time + self.toll_cost
      self.model.counter_traveled_R153 += 1

#%%
class Model_5link_toll(Model_5link):
  """"""
  def __init__(self, N, base_toll_cost, verbose=False):
    super().__init__(N, verbose)
    self.base_toll_cost = base_toll_cost
    self.experienced_travel_time_R153 = None
    self.datacollector = mesa.DataCollector(model_reporters={
        "route_R14": "most_recent_travel_time_R14",
        "route_R23": "most_recent_travel_time_R23",
        "route_R153": "most_recent_travel_time_R153",
        "counter_R14": "counter_traveled_R14",
        "counter_R23": "counter_traveled_R23",
        "counter_R153": "counter_traveled_R153",
        "experienced_R153": "experienced_travel_time_R153"
    })

  #override
  def initialize_agents_positions(self) -> None:
    """Create agents and initialize them to the schedule and infrastructure"""
    for idx in range(self.num_agents):
      car = Car_toll(idx, self)
      # Add the agent to the scheduler
      self.schedule.add(car)
      self.infrastructure[Gl.E_SPAWN]._add_agent(car, idx)