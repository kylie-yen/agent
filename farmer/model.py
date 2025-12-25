from mesa import Model
from mesa.datacollection import DataCollector
import random

from agents import FarmerAgent, GovernmentAgent
from llm_utils import LLMBrain

class VillageModel(Model):
    def __init__(self, subsidy_amount=800):
        super().__init__()
        self.llm_brain = LLMBrain()
        self.steps = 0
        self.subsidy_amount = subsidy_amount
        
        # --- 新增：用于存储农户想法的日志列表 ---
        self.journal = [] 
        
        self.government = GovernmentAgent("Gov", self)
        self.agents.add(self.government)
        
        profiles = [
            {"name": "老王", "character": "富裕、有远见、敢于冒险", "risk": "激进"},
            {"name": "大李", "character": "精明、看重投入产出比、随大流", "risk": "中等"},
            {"name": "小张", "character": "贫穷、极度厌恶亏损、生存第一", "risk": "保守"}
        ]
        
        self.farmers = []
        self.farmers.append(FarmerAgent("Farmer_A", self, profiles[0], 30, 2))
        self.farmers.append(FarmerAgent("Farmer_B", self, profiles[1], 10, 4))
        self.farmers.append(FarmerAgent("Farmer_C", self, profiles[2], 8, 6))
        
        for f in self.farmers:
            self.agents.add(f)
        
        self.datacollector = DataCollector(
            agent_reporters={
                "EcoArea": lambda a: a.eco_crop_area if isinstance(a, FarmerAgent) else None,
                "Wealth": lambda a: a.cumulative_income if isinstance(a, FarmerAgent) else None
            },
            model_reporters={
                "TotalEcoArea": lambda m: sum(f.eco_crop_area for f in m.farmers),
                "AvgWealth": lambda m: sum(f.cumulative_income for f in m.farmers) / len(m.farmers)
            }
        )

    def step(self):
        self.datacollector.collect(self)
        random.shuffle(self.farmers)
        for farmer in self.farmers:
            farmer.step()
        self.steps += 1