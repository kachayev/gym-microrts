"""
Design document: https://www.dropbox.com/scl/fi/3vnzdno2b96u8j8ybhza6/FSP-League-Design.paper?dl=0&rlkey=gcybvvihg3otopkc81aloevkf
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import multiprocessing as mp

num_workers = 2
max_num_matches = 100

class Player:

    def __init__(self, player_id: int):
        self.player_id = player_id
        self.eta = None
        self.rl_policy = None
        self.sl_policy = None
        self.rl_buffer = DequeueBuffer()
        self.sl_buffer = ReservoirBuffer()

    def reset(self):
        self.best_resp = random.random() > self.eta
        # xxx(okachaiev): not sure I need to do it here
        self.rl_buffer.clear()
        self.sl_buffer.clear()

    def get_action(self, obs):
        return self.rl_policy(obs) if self.best_resp else self.sl_policy(obs)

    def log_sample(self, obs, action, reward, next_obs, done):
        self.rl_buffer.push((obs, action, reward, next_obs, done))
        if self.best_resp:
            self.sl_buffer.push((obs, action))

    # xxx(okachaiev): i can move train/log/act into Policy class to avoid
    # repeating the logic of switching all the time
    def train(self):
        pass

    def get_params(self):
        pass

    def set_params(self):
        pass


@dataclass
class Match:
    match_id: int
    p1: Player
    p2: Player

    def create_result(result: int) -> MatchResult:
        return MatchResult(match_id, result)


@dataclass
class MatchResult:
    match_id: int
    result: int


@dataclass
class Payoff:
    num_wins: int = 0
    num_losses: int = 0
    num_draws: int = 0


def league_archive(population, player: Player) -> Player:
    params, = player.get_params()
    mmr, next_id = population.get_mmr(player), population.get_current_generation()
    new_player = Player(player_id=next_id+1, bracket=Player.ARCHIVED, mmr=mmr)
    new_player.set_params(params)
    save_player(new_player)
    population.add_player(new_player)
    return new_player


def league_loop(match_queue, pipe):
    """The League initialization:
    * setups up a `PayoffTable`
    * setups initial population of `Player`(s)
    * generates initial batch of `Match`s

    Note that initialization could be done either from
    scratch or from a given checkpoint (when provided).
    
    The League loop consists of the following duties:
    
    * selects next player to play based on the algorithm provided
    * selects the opponent based on the configuration attached to the player
    * submits new `Match` into the queue
    * on recieving `Match` result from the `Worker` updates payoff table
    * keep track when `Archiving` needs to be performed
    
    In addition to the algorithmical duties, the main process also
    responsible for logging, reporting, checkpointing, etc.
    """
    # xxx(okacahaiev): read this from the file if available
    payoff_table = defaultdict(lambda: defaultdict(Payoff))
    # xxx(okachaiev): read folder with saved models, load them all
    # i can use subfolder structure to deal with different brackets
    population = [Player(player_id=i) for i in range(10)]

    # submit chunk of games to be played
    # xxx(okachaiev): random "all-vs-all" is incredibly inefficient
    init_match_queue = list(combinations(population, 2))
    np.random.shuffle(init_match_queue)
    next_match_id = 0
    for (p1, p2) in init_match_queue:
        match_queue.put(Match(next_match_id, p1, p2))
        next_match_id += 1

    while next_match_id <= max_num_matches:
        match_result = match_result_queue.get()
        # update payoffs
        league_log_match_history(match_result)
        payoff_table[match_result.p1][match_result.p2] += match_result
        payoff_table[match_result.p2][match_result.p1] += (-1)*match_result
        # ... and MMR
        population.update_mmr(p1, p2, match_result)
        population.update_mmr(p2, p1, (-1)*match_result)
        # train if needed
        if match_result.p1.read_to_train():
            match_queue.put(("TRAIN", match_result.p1))
        if match_result.p2.read_to_train():
            match_queue.put(("TRAIN", match_result.p2))
        # archive if needed
        if match_result.p1.ready_to_archive():
            league_archive(population, match_result.p1)
        if match_result.p2.ready_to_archive():
            league_archive(population, match_result.p2)
        # issue a new game to be played
        p1, p2 = league_new_match(population, payoff_table)
        match_queue.put(("MATCH", Match(next_match_id, p1, p2)))
        next_match_id += 1


def setup_env():
    pass


def worker_play(match: Match) -> MatchResult:
    p1 = match.p1.reset()
    p2 = match.p2.reset()
    p1_obs, p2_obs = env.reset()
    episode_len = 0
    for idx in range(1, max_frames+1):
        with torch.no_grad():
            # xxx(okachaiev): torch wrapper? :thinking:
            p1_action = p1.get_action(torch.FloatTensor(p1_obs).to(device)).cpu().numpy()
            p2_action = p2.get_action(torch.FloatTensor(p2_obs).to(device)).cpu().numpy()
        (p1_next_obs, p2_next_obs), reward, done, info = env.step((p1_action, p2_action))
        p1.log_sample(p1_obs, p1_action, reward, p1_next_obs, done)
        p2.log_sample(p2_obs, p2_action, reward, p2_next_obs, done)
        (p1_obs, p2_obs) = (p1_next_obs, p2_next_obs)
        episode_len += 1
        if done or episod_len >= args.episod_episode_len:
            (p1_obs, p2_obs) = env.reset()
            episode_len = 0
    return nmatch.create_result(MatchResult.WIN)

def worker_train(player: Player) -> None:
    pass

def worker_init(queue, pipe):
    """On initialization the worker spawns a new instance of
    the environment to be used for running matches.

    The worker loop consists of the following steps:

    * requests a new `Match` from the `League` (via shared queue)
    * resets the instance of `Environment`
    * setups both `Player`s based on the configuration given
    * plays the game (following Open AI Gym interface)
    * reports the outcome to the `League`
    * if asked to do so, runs either RL or SL training loops
    """
    env = setup_env()
    while True:
        request_type, request_param = queue.get()
        # xxx(okachaiev): QUIT should go to a different channel
        if request_type == "QUIT":
            break
        elif request_type == "MATCH":
            match_result = worker_play(request_param)
            pipe.send(match_result)
        elif request_type == "TRAIN":
            worker_trainn(request_param)


if __name__ == "__main__":
    ctx = mp.get_context('spawn')
    league_conn, worker_conn = ctx.Pipe()
    match_queue = ctx.Queue()
    workers = [ctx.Process(target=worker_init, args=(match_queue, worker_conn,)) for _ in range(num_workers)]
    for w in workers:
        w.start()

    league_loop(match_queue, league_conn)

    for w in workers:
        w.join()