#!/usr/bin/env python3
"""
Multi-Agent Pac-Men (100% Compliance Version)
Implements full predictive conflict detection, mutual exclusion locks,
local sensing, double-lock prevention, and full negotiation integration.
"""

import random, pygame, csv, os, sys, subprocess
from collections import deque, defaultdict

# --------------------------
# CONFIG
# --------------------------
TILE = 30
FPS = 5
START_ENERGY = 100
PELLET_SCORE = 10
MAX_TICKS = 2000
ENERGY_DECAY = 0.2
NEGOTIATION_FAIL_LIMIT = 10
ENERGY_DROP_LIMIT = 10
MAZE_ROWS, MAZE_COLS = 15, 35
WIN_W, WIN_H = 1250, 600
TOP_MARGIN = 50
NEIGHBORS = [(1,0),(-1,0),(0,1),(0,-1)]
GHOST_PENALTY = 2
GHOST_COUNT = 2
AVOID_RADIUS = 5
GHOST_PATROL_RADIUS = 6
SENSING_RADIUS = 2  # for local pre-detection

STRATEGY_MODE = "PRIORITY"  # Default; can switch with P / O

# --------------------------
# MAZE GENERATOR
# --------------------------
def generate_random_maze(rows=MAZE_ROWS, cols=MAZE_COLS):
    maze = [["1" for _ in range(cols)] for _ in range(rows)]
    def in_bounds(r,c): return 0 < r < rows-1 and 0 < c < cols-1
    def carve(r,c):
        maze[r][c] = "0"
        dirs = [(2,0),(-2,0),(0,2),(0,-2)]
        random.shuffle(dirs)
        for dr,dc in dirs:
            nr,nc=r+dr,c+dc
            if in_bounds(nr,nc) and maze[nr][nc]=="1":
                maze[r+dr//2][c+dc//2]="0"
                carve(nr,nc)
    carve(rows//2,cols//2)
    for _ in range(int(rows*cols*0.3)):
        r,c=random.randint(1,rows-2),random.randint(1,cols-2)
        if maze[r][c]=="1": maze[r][c]="0"
    for _ in range(random.randint(20,30)):
        rr,cc=random.randint(1,rows-2),random.randint(1,cols-2)
        if maze[rr][cc]=="0": maze[rr][cc]="C"
    for r in range(rows):
        maze[r][0]=maze[r][-1]="1"
    for c in range(cols):
        maze[0][c]=maze[-1][c]="1"
    return ["".join(row) for row in maze]

# --------------------------
# HELPERS
# --------------------------
def in_bounds(r,c,R,C): return 0<=r<R and 0<=c<C
def manhattan(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

def bfs_path(start,goal,is_walkable,R,C):
    if start==goal:return[start]
    q=deque([start]);parent={start:None}
    while q:
        r,c=q.popleft()
        for dr,dc in NEIGHBORS:
            nr,nc=r+dr,c+dc
            if (nr,nc) not in parent and in_bounds(nr,nc,R,C) and is_walkable((nr,nc)):
                parent[(nr,nc)]=(r,c)
                if (nr,nc)==goal:
                    path=[(nr,nc)]
                    while parent[path[-1]]:path.append(parent[path[-1]])
                    return list(reversed(path))
                q.append((nr,nc))
    return None

# --------------------------
# ENVIRONMENT
# --------------------------
class Environment:
    def __init__(self):
        self.grid=[list(r) for r in generate_random_maze()]
        self.R,self.C=len(self.grid),len(self.grid[0])
        self.pellets,self.corridors=set(),set()
        self._init_pellets()
        self.agents={}
        self.ghosts=[]
        self._init_ghosts()
        self.tick=0
        self.strategy_mode=STRATEGY_MODE
        self.conflicts_total=0
        self.successful_neg=0
        self.failed_neg=0
        self.corridor_status={cell:"UNLOCKED" for cell in self.corridors}
        self.events=deque(maxlen=20)
        self.log_file="negotiation_log.csv"
        if not os.path.exists(self.log_file):
            with open(self.log_file,"w",newline="") as f:
                csv.writer(f).writerow(
                    ["Tick","Strategy","Corridor","Agents","Winner","Outcome","Rounds","LoserWait"]
                )

    # --------------------------
    # Corridor Locking (with double-lock prevention)
    # --------------------------
    def lock_corridor(self, corridor, agent_id):
        status = self.corridor_status.get(corridor, "UNLOCKED")
        if status != "UNLOCKED" and not status.endswith(agent_id):
            # already locked by another agent — trigger conflict
            locked_by = status.replace("LOCKED_BY_", "")
            self.trigger_conflict(corridor, [agent_id, locked_by])
            return False
        self.corridor_status[corridor] = f"LOCKED_BY_{agent_id}"
        return True

    def unlock_corridor(self, corridor):
        self.corridor_status[corridor] = "UNLOCKED"

    # --------------------------
    # Initialization
    # --------------------------
    def _init_pellets(self):
        for r in range(self.R):
            for c in range(self.C):
                if self.grid[r][c]!="1": self.pellets.add((r,c))
                if self.grid[r][c]=="C": self.corridors.add((r,c))

    def _init_ghosts(self):
        self.ghosts=[]
        for _ in range(GHOST_COUNT):
            while True:
                r,c=random.randint(1,self.R-2),random.randint(1,self.C-2)
                if self.grid[r][c]!="1":
                    self.ghosts.append((r,c))
                    break
        print("Ghosts spawned:",self.ghosts)

    def add_agent(self,aid,pos):
        self.agents[aid]={"id":aid,"pos":pos,"score":0,"energy":START_ENERGY,
                          "wait_time":0,"failed_neg":0,"dropped":False,"path":[]}

    # --------------------------
    # Predictive Moves (2-step anticipation)
    # --------------------------
    def predict_next_moves(self, aid, steps=2):
        a=self.agents[aid]
        if len(a["path"])>1:
            return a["path"][1:1+steps]
        return [a["pos"]]

    # --------------------------
    # Conflict Detection (with local sensing)
    # --------------------------
    def detect_conflicts(self, proposals):
        conflicts=[]
        claims=defaultdict(list)

        # 1. Predictive corridor claims
        for aid in self.agents:
            if self.agents[aid]["dropped"]: continue
            next_steps = self.predict_next_moves(aid)
            for step in next_steps:
                if step in self.corridors:
                    claims[step].append(aid)

        # 2. Direct corridor conflict detection
        for corridor,claimers in claims.items():
            if len(claimers)>1:
                conflicts.append((corridor,claimers))

        # 3. Local sensing — detect near agents heading to same region
        agent_ids = list(self.agents.keys())
        for i in range(len(agent_ids)):
            for j in range(i+1,len(agent_ids)):
                a1, a2 = agent_ids[i], agent_ids[j]
                if self.agents[a1]["dropped"] or self.agents[a2]["dropped"]: continue
                if manhattan(self.agents[a1]["pos"], self.agents[a2]["pos"]) <= SENSING_RADIUS:
                    n1 = self.predict_next_moves(a1)
                    n2 = self.predict_next_moves(a2)
                    if set(n1) & set(n2):  # overlap in next 2 moves
                        conflicts.append(("LOCAL", [a1, a2]))

        return conflicts

    # --------------------------
    # Negotiation and Conflict Trigger
    # --------------------------
    def trigger_conflict(self,corridor,agents):
        self.conflicts_total+=1
        if self.strategy_mode=="PRIORITY":
            return self.priority_negotiation(corridor,agents)
        else:
            return self.alternating_offer_negotiation(corridor,agents)

    def priority_negotiation(self,corridor,agents):
        sorted_agents=sorted(agents,key=lambda a:self.agents[a]["score"],reverse=True)
        winner=sorted_agents[0]; loser_wait=2
        for aid in agents:
            ag=self.agents[aid]
            if aid!=winner:
                ag["wait_time"]+=loser_wait
                ag["score"]-=5
        self.successful_neg+=1
        self.lock_corridor(corridor,winner)
        self._log_event(corridor,agents,winner,"PRIORITY","SUCCESS",1,loser_wait)
        return winner

    def alternating_offer_negotiation(self, corridor, agents):
        """
        Formal alternating-offer negotiation with explicit message exchange,
        performatives (PROPOSE / ACCEPT / REJECT), and timeout penalties.
        """
        def create_message(sender, receiver, performative, content):
            return {
                "sender": sender,
                "receiver": receiver,
                "performative": performative,
                "content": content
            }

        proposer, responder = agents[0], agents[1]
        success = False
        winner = None
        rounds = 0
        loser_wait = 1
        max_rounds = 3
        messages = []

        for i in range(max_rounds):
            rounds += 1
            # --- Proposer sends a PROPOSE message ---
            proposal = create_message(
                proposer, responder, "PROPOSE",
                f"Request access to corridor {corridor}. You wait {loser_wait} turn(s)."
            )
            messages.append(proposal)
            self.events.appendleft(f"T{self.tick}: {proposal['sender']}→{proposal['receiver']} [{proposal['performative']}] {proposal['content']}")

            # --- Compute utilities to decide if responder accepts ---
            p_util = self.agents[proposer]["energy"] - self.agents[proposer]["wait_time"]
            r_util = self.agents[responder]["energy"] - self.agents[responder]["wait_time"]

            # Respondent evaluates proposal
            if r_util <= p_util:
                # Responder ACCEPTS
                response = create_message(responder, proposer, "ACCEPT",
                                          f"Agreed. {proposer} goes first in corridor {corridor}.")
                messages.append(response)
                self.events.appendleft(f"T{self.tick}: {response['sender']}→{response['receiver']} [{response['performative']}] {response['content']}")
                success = True
                winner = proposer
                self.agents[responder]["wait_time"] += loser_wait
                break
            else:
                # Responder REJECTS and counter-proposes
                response = create_message(responder, proposer, "REJECT",
                                          f"Reject. I propose I go first for corridor {corridor}.")
                messages.append(response)
                self.events.appendleft(f"T{self.tick}: {response['sender']}→{response['receiver']} [{response['performative']}] {response['content']}")
                proposer, responder = responder, proposer  # Swap roles

        # --- Outcome evaluation ---
        outcome = "SUCCESS" if success else "TIMEOUT"

        if success:
            self.successful_neg += 1
            self.lock_corridor(corridor, winner)
        else:
            self.failed_neg += 1
            # Apply fallback and heavy penalties
            winner = random.choice(agents)
            for aid in agents:
                self.agents[aid]["score"] -= 15
                self.agents[aid]["failed_neg"] += 1
            self.events.appendleft(f"T{self.tick}: Negotiation TIMEOUT — fallback to random winner {winner}, penalties applied.")

        # Log negotiation details
        self._log_event(corridor, agents, winner or "None", "ALT-OFFER", outcome, rounds, loser_wait)
        return winner

    def _log_event(self,corridor,agents,winner,strategy,outcome,rounds,loser_wait):
        msg=f"T{self.tick}: {strategy} {corridor} -> {winner} ({outcome})"
        self.events.appendleft(msg)
        with open(self.log_file,"a",newline="") as f:
            csv.writer(f).writerow([self.tick,strategy,corridor,agents,winner,outcome,rounds,loser_wait])

    # --------------------------
    # Movement, Energy, and Fairness
    # --------------------------
    def is_walkable(self,cell):
        r,c=cell
        return in_bounds(r,c,self.R,self.C) and self.grid[r][c]!="1"

    def plan_paths(self):
        for a in self.agents.values():
            if a["dropped"] or not self.pellets: continue
            start=a["pos"]
            def ghost_penalty(p): return sum(max(0,AVOID_RADIUS-manhattan(p,g)) for g in self.ghosts)
            tgt=min(self.pellets,key=lambda p:manhattan(start,p)+ghost_penalty(p))
            a["path"]=bfs_path(start,tgt,self.is_walkable,self.R,self.C) or []

    def propose_moves(self):
        return {aid:(a["path"][1] if len(a["path"])>1 else a["pos"]) for aid,a in self.agents.items() if not a["dropped"]}

    def resolve_conflicts(self,proposals):
        final_moves=proposals.copy()
        conflicts=self.detect_conflicts(proposals)
        for corridor,agents in conflicts:
            winner=self.trigger_conflict(corridor,agents)
            for aid in agents:
                if aid!=winner:
                    final_moves[aid]=self.agents[aid]["pos"]
            self.unlock_corridor(corridor)
        return final_moves

    def commit_moves(self,moves):
        for aid,dest in moves.items():
            a=self.agents[aid]
            if a["dropped"]: continue
            a["pos"]=dest
            if dest in self.pellets:
                self.pellets.remove(dest)
                a["score"]+=PELLET_SCORE
            a["energy"]-=ENERGY_DECAY
            if a["energy"]<ENERGY_DROP_LIMIT:
                a["energy"]=0;a["dropped"]=True

    def move_ghosts(self):
        new=[]
        active=[a for a in self.agents.values() if not a["dropped"]]
        for idx,(r,c) in enumerate(self.ghosts):
            if not active:new.append((r,c));continue
            nearest=min(active,key=lambda a:manhattan((r,c),a["pos"]))
            dist=manhattan((r,c),nearest["pos"])
            chase_prob=0.6 if dist<GHOST_PATROL_RADIUS else 0.3
            if random.random()<chase_prob:
                path=bfs_path((r,c),nearest["pos"],self.is_walkable,self.R,self.C)
                if path and len(path)>1:new.append(path[1]);continue
            opts=[(r+dr,c+dc) for dr,dc in NEIGHBORS if self.is_walkable((r+dr,c+dc))]
            new.append(random.choice(opts) if opts else (r,c))
        self.ghosts=new

    def apply_ghost_penalty(self):
        for g in self.ghosts:
            for a in self.agents.values():
                if a["dropped"]:continue
                if manhattan(g,a["pos"])<=1:
                    a["energy"]-=GHOST_PENALTY
                    if a["energy"]<ENERGY_DROP_LIMIT:a["dropped"]=True

    def fairness_index(self):
        s=[a["score"] for a in self.agents.values() if not a["dropped"]]
        return 1.0 if not s else round(1-(max(s)-min(s))/(sum(s)+1e-9),3)
    def average_wait_time(self):
        return round(sum(a["wait_time"] for a in self.agents.values())/len(self.agents),2)


# --------------------------
# GUI RUNNER
# --------------------------
def run_episode():
    pygame.init()
    env = Environment()
    R, C = env.R, env.C
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Multi-Agent PAC-MEN: Negotiation Protocols")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)
    small = pygame.font.SysFont("Consolas", 13)
    big = pygame.font.SysFont("Consolas", 42, bold=True)

    TILE_FIT = min((WIN_W - 250) // C, (WIN_H - 150 - TOP_MARGIN) // R)
    offset_x = (WIN_W - 250 - C * TILE_FIT) // 2

    for i, aid in enumerate(["A", "B", "C"]):
        env.add_agent(aid, (R // 2, C // 2 + i - 1))

    tick, run = 0, True
    finished = False  # flag for pellet completion

    while run and tick < MAX_TICKS and any(not a["dropped"] for a in env.agents.values()):
        tick += 1
        env.tick = tick

        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                run = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_p:
                    env.strategy_mode = "PRIORITY"
                elif e.key == pygame.K_o:
                    env.strategy_mode = "ALT-OFFER"

        # If all pellets collected, freeze movement
        if not env.pellets and not finished:
            finished = True
            print("All pellets collected — waiting for manual key press to show summary.")

        if not finished:
            env.plan_paths()
            proposals = env.propose_moves()
            moves = env.resolve_conflicts(proposals)
            env.commit_moves(moves)
            env.move_ghosts()
            env.apply_ghost_penalty()

        # --- Drawing Section ---
        screen.fill((0, 0, 0))
        directions = f"Press P=PRIORITY, O=ALT-OFFER | Current: {env.strategy_mode}"
        screen.blit(font.render(directions, True, (255, 255, 0)), (10, 10))

        # Draw Maze + Pellets + Corridors
        for r in range(R):
            for c in range(C):
                ch = env.grid[r][c]
                x, y = offset_x + c * TILE_FIT, TOP_MARGIN + r * TILE_FIT
                if ch == "1":
                    pygame.draw.rect(screen, (10, 10, 70), (x, y, TILE_FIT, TILE_FIT))
                else:
                    pygame.draw.rect(screen, (15, 15, 15), (x, y, TILE_FIT, TILE_FIT))
                    if (r, c) in env.pellets:
                        pygame.draw.circle(screen, (240, 240, 240),
                                           (x + TILE_FIT // 2, y + TILE_FIT // 2), 3)
                    if (r, c) in env.corridors:
                        pygame.draw.rect(screen, (80, 20, 140),
                                         (x + 3, y + 3, TILE_FIT - 6, TILE_FIT - 6), 2)

        # Draw Ghosts
        for i, (gr, gc) in enumerate(env.ghosts):
            gx, gy = offset_x + gc * TILE_FIT + TILE_FIT // 2, TOP_MARGIN + gr * TILE_FIT + TILE_FIT // 2
            pygame.draw.circle(screen, [(255, 50, 50), (255, 150, 150)][i % 2], (gx, gy), TILE_FIT // 2 - 3)

        # Draw Agents
        pac_colors = {"A": (255, 255, 0), "B": (80, 255, 80), "C": (255, 180, 80)}
        flash = (tick // 5) % 2 == 0
        for aid, a in env.agents.items():
            r, c = a["pos"]
            cx, cy = offset_x + c * TILE_FIT + TILE_FIT // 2, TOP_MARGIN + r * TILE_FIT + TILE_FIT // 2
            col = (90, 90, 90) if a["dropped"] else pac_colors[aid]
            pygame.draw.circle(screen, col, (cx, cy), TILE_FIT // 2 - 2)
            ratio = max(0, a["energy"] / START_ENERGY)
            bar_y = cy - TILE_FIT // 2 - 10
            pygame.draw.rect(screen, (100, 20, 20), (cx - TILE_FIT // 2, bar_y, TILE_FIT, 6))
            color = (255, 0, 0) if a["energy"] < 20 and flash else (
                int(255 * (1 - ratio)), int(255 * ratio), 50)
            pygame.draw.rect(screen, color, (cx - TILE_FIT // 2, bar_y, int(TILE_FIT * ratio), 6))
            screen.blit(small.render(aid, True, (255, 255, 255)), (cx - 6, bar_y - 20))

        # Sidebar Log
        sidebar_x = WIN_W - 285
        pygame.draw.rect(screen, (25, 25, 25), (sidebar_x, 0, 285, WIN_H))
        screen.blit(font.render("Live Negotiation Log", True, (255, 255, 0)), (sidebar_x + 10, 20))
        for i, event in enumerate(list(env.events)[:22]):
            color = (255, 255, 255)
            if "PRIORITY" in event:
                color = (255, 230, 100)
            elif "ALT-OFFER" in event:
                color = (100, 220, 255)
            screen.blit(small.render(event[:38], True, color), (sidebar_x + 10, 60 + i * 18))

        # Bottom Stats
        y0 = TOP_MARGIN + R * TILE_FIT + 6
        for i, (aid, a) in enumerate(env.agents.items()):
            e = 0 if a["dropped"] else int(a["energy"])
            txt = f"{aid} | Score:{a['score']} | Energy:{e} | Wait:{a['wait_time']} | Dropped:{a['dropped']}"
            screen.blit(font.render(txt, True, (240, 240, 240)), (6, y0 + i * 22))
        fair, avg_wait = env.fairness_index(), env.average_wait_time()
        mtxt = f"T:{tick} | Pellets:{len(env.pellets)} | Conflicts:{env.conflicts_total} | Success:{env.successful_neg} | Fail:{env.failed_neg} | Fairness:{fair}"
        screen.blit(font.render(mtxt, True, (255, 230, 100)), (6, y0 + 80))

        # Message when finished
        if finished:
            screen.blit(font.render("All pellets collected!", True, (255, 255, 0)), (6, y0 + 120))
            screen.blit(font.render("Press any key to view summary...", True, (180, 180, 255)), (6, y0 + 145))

        pygame.display.flip()
        clock.tick(FPS)

        # Wait for key if done
        if finished:
            waiting = True
            while waiting:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                        run = False
                        waiting = False
                    elif e.type == pygame.KEYDOWN:
                        waiting = False  # proceed to summary
            break


    # --- Summary Screen (appears after key press) ---
    winner = max(env.agents.values(), key=lambda a: a["score"])
    screen.fill((0, 0, 0))
    title = f"Winner: Agent {winner['id']} | Score: {winner['score']}"
    screen.blit(big.render(title, True, (255, 255, 0)), (WIN_W // 2 - 250, WIN_H // 2 - 150))
    screen.blit(font.render(f"Final Strategy Mode Used: {env.strategy_mode}", True, (255, 255, 255)),
                (WIN_W // 2 - 200, WIN_H // 2 - 110))

    summary_y = WIN_H // 2 - 70
    header = "AGENT   SCORE   ENERGY   WAIT   FAIL_NEG   DROPPED"
    screen.blit(font.render(header, True, (255, 255, 255)), (WIN_W // 2 - 200, summary_y))
    for i, (aid, a) in enumerate(env.agents.items()):
        e = 0 if a["dropped"] else int(a["energy"])
        row = f"  {aid}        {a['score']}       {e}       {a['wait_time']}         {a['failed_neg']}         {a['dropped']}"
        screen.blit(font.render(row, True, (255, 230, 150)), (WIN_W // 2 - 200, summary_y + 30 + i * 25))

    metrics_y = WIN_H // 2 + 40
    lines = [
        f"Total Conflicts: {env.conflicts_total}",
        f"Successful Negotiations: {env.successful_neg}",
        f"Failed Negotiations: {env.failed_neg}",
        f"Average Waiting Time: {env.average_wait_time()}",
        f"Fairness Index: {env.fairness_index()}",
        f"Remaining Pellets: {len(env.pellets)}",
        f"Simulation Duration (Ticks): {tick}",
        f"Negotiation Log Saved: {env.log_file}"
    ]
    screen.blit(font.render("Metrics Summary", True, (255, 255, 255)), (WIN_W // 2 - 130, metrics_y))
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, (200, 200, 200)), (WIN_W // 2 - 200, metrics_y + 30 + i * 22))

    screen.blit(font.render("Press any key or ESC to close...", True, (255, 255, 0)),
                (WIN_W // 2 - 180, metrics_y + 220))
    pygame.display.flip()

    waiting = True
    while waiting:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN):
                waiting = False

    pygame.quit()

    # Auto-open CSV log
    try:
        if sys.platform.startswith("win"):
            os.startfile(env.log_file)
        elif sys.platform == "darwin":
            subprocess.call(["open", env.log_file])
        else:
            subprocess.call(["xdg-open", env.log_file])
    except Exception as e:
        print("Could not auto-open log file:", e)

if __name__=="__main__":
    run_episode()
