import random
import math
import sys
import numpy as np
from copy import deepcopy
from typing import Iterator, Callable, TypeVar, NamedTuple, Optional
from typing_extensions import Self


class Factory:
	class Truck(NamedTuple):
		capacity: int
		max_distance: int
		distance_cost: int
		day_cost: int
		cost: int
	class TechCost(NamedTuple):
		distance_cost: int
		day_cost: int
		cost: int
	class Machine(NamedTuple):
		id: int
		weight: int
		penalty: int
	class Location(NamedTuple):
		id: int
		x: int
		y: int
	class Request(NamedTuple):
		id: int
		location: int
		first: int
		last: int
		machine_id: int
		machine_quantity: int
	class Technician(NamedTuple):
		id: int
		location: int
		dist_limit: int
		req_limit: int
		machine: int

	days: int
	truck: Truck
	tech_cost: TechCost
	machines: list[Machine]
	locations: list[Location]
	requests: list[Request]
	technicians: list[Technician]
	hard_penalty: int

	def __init__(
			self, fname: str,
			hard_penalty: int | None = None, seed: int | None = None
		):
		self.machines = []
		self.locations = []
		self.requests = []
		self.technicians = []
		T = TypeVar('T', bound=NamedTuple)
		truck = {
			"truck_capacity": 0, "truck_max_distance": 0,
			"truck_distance_cost": 0, "truck_day_cost": 0, "truck_cost": 0
		}
		tech_cost = {
			"technician_distance_cost": 0,
			"technician_day_cost": 0, "technician_cost": 0
		}
		mapping = {
			"machines": (self.machines, self.Machine),
			"locations": (self.locations, self.Location),
			"requests": (self.requests, self.Request)
		}
		def sort(src: list[T]) -> list[T]:
			return sorted(src, key=lambda src: src.id)

		def parse_map(
				dst: list[T], ref: list[T],
				start: int, no: int, lines: list[str]
			):
			"""
			Generic function that parses a space-separated line
			into a namedtuple based on the provided template.
			"""
			count = 0
			for line in lines[start:]:
				if count == no:
					break
				values = list(map(int, line.split()))
				dst.append(ref(*values))
				count += 1

		def parse_tech(
				dst: list[T], ref: list[T],
				start:int, no: int, lines: list[str]
			):
			count = 0
			for line in lines[start:]:
				if count == no:
					break
				values = list(map(int, line.split()))
				buffer = values[4:]
				values = values[:5]
				bits = sum(j<<i for i,j in enumerate(buffer))
				values[4] = bits
				dst.append(ref(*values))
				count += 1

		with open(fname, 'r') as f:
			lines = f.readlines()
			i = 0
			while i < len(lines):
				line = lines[i].strip()
				i += 1
				if len(line) == 0:
					continue
				key, no = line.split("=")
				key = key.strip().lower()
				no = int(no.strip())
				if key == "days":
					self.days = int(no)
				elif key in truck:
					truck[key] = no
				elif key in tech_cost:
					tech_cost[key] = no
				elif key in mapping:
					dst, ref = mapping[key]
					parse_map(dst, ref, i, no, lines)
					i += no
				else:
					parse_tech(self.technicians, self.Technician, i, no, lines)
					i += no
		self.machines = sort(self.machines)
		self.locations = sort(self.locations)
		self.requests = sort(self.requests)
		self.technicians = sort(self.technicians)
		self.truck = self.Truck(*truck.values())
		self.tech_cost = self.TechCost(*tech_cost.values())
		if hard_penalty is None:
			hard_penalty = (max(truck.values()) + max(tech_cost.values())) ** 2
		self.hard_penalty = hard_penalty
		if isinstance(seed, int):
			random.seed(seed)

	def __str__(self) -> str:
		s = f"DAYS: {self.days}\n"
		s += str(self.truck) + '\n'
		s += str(self.tech_cost) + '\n'
		s += "machines\n"
		for mach in self.machines:
			s += str(mach) + '\n'
		s += "locations\n"
		for loc in self.locations:
			s += str(loc) + '\n'
		s += "requests\n"
		for req in self.requests:
			s += str(req) + '\n'
		s += "technician\n"
		for tech in self.technicians:
			s += str(tech) + '\n'
		return s
	def getLoc(self, id: int) -> tuple[int]:
		return (self.locations[id - 1].x, self.locations[id - 1].y)

	def getCap(self, id: int) -> int:
		req = self.requests[id - 1]
		size = self.machines[req.machine_id - 1].weight
		return size * req.machine_quantity

	def checkTechSkill(self, tech_id: int, machine_id: int) -> bool:
		bits = self.technicians[tech_id - 1].machine
		return (bits & (1 << machine_id - 1)) != 0

	@staticmethod
	def getDist(pt1: tuple[int, int], pt2: tuple[int, int]) -> float:
		x1, y1 = pt1; x2, y2 = pt2
		return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def getDist(pt1: tuple[int, int], pt2: tuple[int, int]) -> float:
	x1, y1 = pt1; x2, y2 = pt2
	return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def skewed_sample(start, end, scale=5):
	value = int(np.random.exponential(scale)) + start
	return min(value, end)
