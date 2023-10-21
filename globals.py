class Gl():
    # structure names: junctions and edges
    J1 = "j1"
    J2 = "j2"
    J3 = "j3"
    J4 = "j4"
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    E4 = "E4"
    E5 = "E5"
    E0 = "E0"
    E_SPAWN = "E_spawn"
    E_SPAWN_BUS = "E_spawn_bus"

    # routes
    R14 = "R14"
    R23 = "R23"
    R153 = "R153"
    ROUTES_4LINK = [R14, R23]
    ROUTES_5LINK = [R14, R23, R153]

    # hyperparameters
    """mostly used for the agent's memory-based strategy on choosing routes"""
    DELTA_T = 10
    PROB_INFO = 0.9
    J1_WAIT_THRESHOLD = 0.1
    J2_WAIT_THRESHOLD = 0.1
