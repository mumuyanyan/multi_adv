import numpy as np
from env.multiagent.core_noise import World, Agent, Landmark
from env.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        
        world.dim_c = 2
        num_good_agents = 2
        num_adversaries = 4
        num_obstacles = 0

        num_agents = num_adversaries + num_good_agents
        num_landmarks = 2
        world.num_agents = num_agents
        world.num_good_agents = num_good_agents
        world.num_adversaries = num_adversaries
        world.num_landmarks = num_landmarks
        size_adv = 0.025
        size_good = 0.04
        size_landmark = 0.05
        world.size_adv_good = size_adv + size_good
        world.size_landmark_good = size_good + size_landmark
        world.good_size = size_good
        world.adv_size = size_adv
        world.land_size = size_landmark
        
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.id = i
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = False if i < num_good_agents else True
            agent.size = size_adv if agent.adversary else size_good
            agent.accel = 2.0 if agent.adversary else 4.0
            
            agent.max_speed = 0.5 if agent.adversary else 1.3
            agent.max_speed = 0.3 if agent.adversary else 1.
            agent.max_speed = 0.5 if agent.adversary else 1.

        
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.id = i
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = size_landmark
            landmark.boundary = False

        world.obstacles = [Landmark() for i in range(num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.id = i
            obstacle.name = 'obstacle %d' % i
            obstacle.collide = True
            obstacle.movable = False
            
            obstacle.size = 0.15
            obstacle.boundary = False
        
        self.reset_world(world)
        return world


    def reset_world(self, world,seed=None):
        if seed != None:
            np.random.seed(seed)
        
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35- i*0.5, 0.85- i*0.5, 0.35- i*0.5]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25- i*0.15, 0.25- i*0.15, 0.25- i*0.15])

        for i, obstacle in enumerate(world.obstacles):
            
            obstacle.color = np.array([0.35, 0.35, 0.85])
        
        pos_lis = []

        obstacles = [[0., 0.], [0.5, 0.5], [-0.5, -0.5]]
        for i, obstacle in enumerate(world.obstacles):
            if not obstacle.boundary:
                
                obstacle.state.p_pos = np.array(obstacles[i])

                obstacle.state.p_vel = np.zeros(world.dim_p)

            
        landmark_pos = [np.array([0.,0.])]
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                flag = True
                while(flag):
                    count = 0
                    landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                    for i, obstacle in enumerate(world.obstacles):
                        if not self.is_collision_aug(obstacle, landmark,2.):
                            count = count + 1
                    landmark_c = [np.linalg.norm(landmark.state.p_pos-landmark_pos[j]) > landmark.size*17 for j in range(len(landmark_pos))]
                    if count == len(world.obstacles) and not(False in landmark_c):
                        flag = False
                        landmark_pos.append(landmark.state.p_pos)
            
            landmark.state.p_vel = np.zeros(world.dim_p)


        agent_pos = [np.array([0.,0.])]
        for agent in world.agents:
            flag = True
            while(flag):
                count = 0
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)

                for i, obstacle in enumerate(world.obstacles):
                    if not self.is_collision_aug(obstacle, agent,2.):
                        count = count + 1
                landmark_c = [np.linalg.norm(agent.state.p_pos - landmark_pos[j]) > agent.size * 6 for j in
                              range(len(landmark_pos))]
                agent_c = [np.linalg.norm(agent.state.p_pos - agent_pos[j]) > agent.size * 15 for j in
                              range(len(agent_pos))]

                if count == len(world.obstacles) and not(False in landmark_c) and not(False in agent_c):
                    flag = False
                    agent_pos.append(agent.state.p_pos)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.u_noise = True
            agent.step = 0


    def constrained_log_cbf_bound(self,hx, epsion, gamma,rho):
        if hx-epsion>0:
            return -np.log(gamma * (hx - epsion + rho) / (gamma*(hx-epsion+rho) + 1))
        else:
            hx = epsion
            return -np.log(gamma * (hx - epsion + rho) / (gamma*(hx-epsion+rho) + 1))

    def constrained_log_cbf(self,hx, epsion, gamma,rho):
            return -np.log(gamma * (hx - epsion + rho) / (gamma*(hx-epsion+rho) + 1))


    def benchmark_data(self, agent, world):
        
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    def is_collision_aug(self, agent1, agent2,rate=1.2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < rate*dist_min else False

    
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        
        
        main_reward = self.agent_reward(agent, world)

        return main_reward

    def cost_reward(self,agent,world,rate=1.2):
        main_rewrd = self.adversary_cost(agent,world,rate) if agent.adversary else self.agent_cost(agent,world,rate)
        return main_rewrd

    def utility_reward(self, agent, world):
        
        
        main_reward = self.agent_utility_reward(agent, world)

        return main_reward


    def safe_reward(self, agent, world):
        
        
        main_reward = self.agent_safe_reward(agent, world)

        return main_reward

    def agent_reward(self, agent, world):
        
        rew = 0
        shape = False
        shape = True
        obstacles = world.obstacles

        landmarks = world.landmarks

        
        
        
        
        
        
        

        '''
        if agent.collide:
            for a in obstacles:
                hx = np.sqrt(np.sum(np.square(agent.state.p_pos - a.state.p_pos))) - 2*(agent.size + a.size)

                if self.is_collision_aug(a, agent):

                    rew -= self.constrained_log_cbf(0,0,10,1e-50)
                else:
                    rew -= self.constrained_log_cbf(hx,0,10,1e-50)
        '''

        for a in landmarks:
            if agent.id == a.id:
                if np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) <= agent.size + a.size:
                    rew += 100
                
                
                rew -= np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos)))
                rew += np.exp(-0.5*(np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos)))-agent.size - a.size))




            
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def agent_utility_reward(self, agent, world):
        
        rew = 0
        shape = False
        shape = True
        obstacles = world.obstacles

        landmarks = world.landmarks
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        for a in landmarks:
            if np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) < 2 * a.size:
                rew += 100
            rew -= np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos)))

            

        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 20)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def agent_safe_reward(self, agent, world):
        
        rew = 0
        shape = False
        shape = True
        obstacles = world.obstacles

        landmarks = world.landmarks
        
        
        
        
        
        
        
        if agent.collide:
            for a in obstacles:
                hx = np.sqrt(np.sum(np.square(agent.state.p_pos - a.state.p_pos))) - 1.2 * (agent.size + a.size)

                if self.is_collision_aug(a, agent):

                    rew -= self.constrained_log_cbf(0, 0, 10, 1e-50)
                else:
                    rew -= self.constrained_log_cbf(hx, 0, 10, 1e-50)
        
        
        
        

            

        
        
        
        
        
        
        
        
        
        

        return rew

    def agent_cost(self, agent, world, rate=1.2):
        safety_cost = 0.0
        safety_costs = []

        obstacles = world.obstacles
        agents = world.agents
        
        
        
        

        for a in obstacles:
            hx =  rate * (agent.size + a.size) - np.sqrt(np.sum(np.square(agent.state.p_pos - a.state.p_pos)))
            if hx > 0:
                safety_cost += hx/(1.2 * (agent.size + a.size))
            safety_costs.append(hx)
        for a in agents:
            if a is agent: continue
            hx =  rate * (agent.size + a.size) - np.sqrt(np.sum(np.square(agent.state.p_pos - a.state.p_pos)))
            if hx > 0:
                safety_cost += hx / (1.2 * (agent.size + a.size))
            safety_costs.append(hx)

        

        cost = float(safety_cost>0.0)
        return safety_cost,cost

    def adversary_cost(self, agent, world, rate=1.2):
        safety_cost = 0.0
        safety_costs = []

        agents = world.agents

        for a in agents:
            if a.adversary: continue
            hx =  rate * (agent.size + a.size) - np.sqrt(np.sum(np.square(agent.state.p_pos - a.state.p_pos)))
            safety_costs.append(hx)

        safety_cost = max(safety_costs)

        cost = float(safety_cost>0.0)
        return safety_cost,cost

    def done(self,agent,world):
        return False


    def adversary_reward(self, agent, world):
        
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def observation(self, agent, world):
        
        entity_pos = []
        entity_vel = []

        if agent.id < world.num_good_agents:
            for entity in world.landmarks:
                if not entity.boundary:
                    if agent.id == entity.id:
                        entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        else:
            entity_pos.append(-agent.state.p_pos)


        for entity in world.agents:
            if entity is agent: continue
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            entity_vel.append(entity.state.p_vel - agent.state.p_vel)

        for entity in world.obstacles:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + entity_vel)
