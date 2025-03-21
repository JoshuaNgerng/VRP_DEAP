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

	class DaySchedule:
		truck: list[int]
		technicians: list[list[int]]
		def __init__(self, truck_part: list[int], tech_part: list[list[int]]):
			self.truck = truck_part.copy(); self.technicians = tech_part.copy()
		def __str__(self) -> str:
			s = "truck schedule\n"
			s += f'{str(self.truck)}\n'
			s += "technicians\n"
			for idx, individual in enumerate(self.technicians):
				if len(individual) == 0:
					continue
				s += f'{idx + 1} -> {individual}\n'
			return s
		def __iter__(self) -> Iterator[list[int] | list[list[int]]]:
				yield self.truck
				yield self.technicians
	class Schedule:
		day_schedules: list['Factory.DaySchedule'] = []
		def __init__(self, truck_part: list[list[int]], tech_part: list[list[list[int]]]):
			self.day_schedules = [
				Factory.DaySchedule(truck_day, tech_day)
				for truck_day, tech_day in zip(truck_part, tech_part)
			]
		def __str__(self) -> str:
			s = "Schedule\n"
			for day_count, day_schedule in enumerate(self.day_schedules):
				s += f"{day_count + 1}\n{day_schedule}\n"
			return s
		def __iter__(self) -> Iterator['Factory.DaySchedule']:
			for day_schedule in self.day_schedules:
				yield day_schedule
		def __getitem__(self, idx: int) -> 'Factory.DaySchedule':
			return self.day_schedules[idx]
		def __len__(self) -> int:
			return len(self.day_schedules)
		def add(self, other: 'Factory.DaySchedule'):
			self.day_schedules.append(other)

	class TruckEvalSheet:
		total_truck_dist: float = 0.0
		max_truck: int = 0
		total_truck: int = 0
		truck_violation: int = 0
		total_cost: int = 0
		penalty: int = 0
		schedule = []
		def __str__(self) -> str:
			s = str(self.__class__.__name__) + '\n'
			s += '\n'.join(f'{key}={value}' for key, value in self.__dict__.items())
			return s
		def add_stat(self, other: Self):
			self.total_truck_dist += other.total_truck_dist
			self.total_truck += other.total_truck
			if self.max_truck < other.max_truck:
				self.max_truck = other.max_truck
			self.truck_violation += other.truck_violation
			self.penalty += other.penalty
			self.total_cost += other.total_cost
		def add(self, other: Self):
			self.add_stat(other)
			if len(other.schedule) > 0:
				self.schedule.append(other.schedule)
		def extend(self, other: Self):
			self.schedule.extend(other.schedule)
		def cal_cost(self, factory: 'Factory') -> int:
			self.total_cost = self.total_truck_dist * factory.truck.distance_cost + \
				self.total_truck * factory.truck.day_cost + \
				self.max_truck * factory.truck.cost + self.penalty
			self.total_cost = int(self.total_cost)
			return self.total_cost
	class TechEvalSheet:
		total_tech_dist: int = 0
		no_tech_employed: int = 0
		total_tech_deployed: int = 0
		idle_machine_cost: int = 0
		total_tech_penalty: int = 0
		tech_violation: list[int] = []
		tech_penalty: list[int] = []
		total_cost: int = 0
		schedule = []
		def __init__(self, length: int = 0):
			self.tech_violation = [0] * length
			self.tech_penalty = [0] * length
		def __str__(self) -> str:
			s = str(self.__class__.__name__) + '\n'
			s += '\n'.join(f'{key}={self.format_value(key, value)}' for key, value in self.__dict__.items())
			return s
		def format_value(self, key: str, value: int | list[int]):
			if key == "tech_violation" or key == "tech_penalty":
				return value.count(1)
			if key == "no_tech_employed":
				return self.totalTechResigter()
			return value
		def add(self, other: Self):
			self.total_tech_dist += other.total_tech_dist
			self.no_tech_employed |= other.no_tech_employed
			self.total_tech_deployed += other.total_tech_deployed
			self.idle_machine_cost += other.idle_machine_cost
			self.total_tech_penalty += other.total_tech_penalty
			if len(self.tech_violation) == 0:
				self.tech_violation = other.tech_violation.copy()
				self.tech_penalty = other.tech_penalty.copy()
				return
			for i in range(len(self.tech_violation)):
				self.tech_violation[i] += other.tech_violation[i]
				self.tech_penalty[i] += other.tech_penalty[i]
		def cal_cost(self, factory: 'Factory') -> int:
			self.total_cost = self.total_tech_dist * factory.tech_cost.distance_cost + \
				self.total_tech_deployed * factory.tech_cost.day_cost + \
				self.totalTechResigter() * factory.tech_cost.cost + \
				self.idle_machine_cost + self.total_tech_penalty
			self.total_cost = int(self.total_cost)
			return self.total_cost
		def resigterTech(self, id: int):
			self.no_tech_employed |= 1 << id
		def checkTechStatus(self, id: int) -> bool:
			return (self.no_tech_employed & (1 << id)) != 0
		def totalTechResigter(self) -> int:
			return bin(self.no_tech_employed).count('1')
	class EvalSheet:
		schedule = []
		def __init__(
				self, truck: Optional['Factory.TruckEvalSheet'] = None, 
				tech: Optional['Factory.TechEvalSheet'] = None
			):
			self.truck = Factory.TruckEvalSheet()
			if truck is not None:
				self.truck.add_stat(truck)
				self.truck.extend(truck)
			self.tech = Factory.TechEvalSheet()
			if tech is not None:
				self.tech.add(tech)
		def __str__(self) -> str:
			return f'{self.__class__.__name__}({", ".join(f"{key}={value}" for key, value in self.__dict__.items())})'
		def cal_cost(self, factory: 'Factory'):
			if self.truck.total_cost == 0:
				self.truck.cal_cost(factory)
			if self.tech.total_cost == 0:
				self.tech.cal_cost(factory)
			return self.truck.total_cost + self.tech.total_cost

	class TechStat:
		dist: int = 0
		dist_limit: int = 0
		worked: bool = False
		deploy_count: int = 0
		deploy_limit: int = 0
		req_count: int = 0
		req_limit: int = 0
		rest_count: int = 0
		rest_day: int = 0
		full_work_shift: bool = False
		pos: tuple[int, int] = (0, 0)
		home: tuple[int, int] = (0, 0)
		def __init__(
				self, start: tuple[int, int],
				dist_limit: int, req_limit: int,
				deploy_limit: int, rest_day: int
			):
			self.pos = start; self.home = start
			self.dist_limit = dist_limit; self.req_limit = req_limit
			self.deploy_limit = deploy_limit; self.rest_day = rest_day
		def reset(self):
			self.dist = 0; self.req_count = 0; self.pos = self.home
			if self.full_work_shift == False and \
				self.deploy_count >= self.deploy_limit:
				self.full_work_shift = True
			if self.worked == True:
				self.deploy_count += 1
				self.rest_count = 0
			else:
				self.deploy_count = 0
				self.rest_count += 1
			if self.rest_count >= self.rest_day:
				self.full_work_shift = False
			self.worked = False
		def avaliable(self) -> bool:
			if self.deploy_count > self.deploy_limit or self.req_count > self.req_count:
				return False
			if self.full_work_shift == True:
				return False
			return True


	days: int
	truck: Truck
	tech_cost: TechCost
	machines: list[Machine]
	locations: list[Location]
	requests: list[Request]
	technicians: list[Technician]
	hard_penalty: int
	individual_creator = None
	truck_schedule_ref = None

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

	def resigterTruckScheduleRef(self, truck: list[int]):
		self.truck_schedule_ref = self.splitScheduleByDay(truck)

	def resigterIndividualClass(self, individual: Callable[[], list[int]]):
		self.individual_creator = individual

	def makeCreatorTemplate(self) -> list[int] | Schedule:
		return self.individual_creator()

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

	@staticmethod
	def skewed_sample(start, end, scale=5):
		value = int(np.random.exponential(scale)) + start
		return min(value, end)

	@staticmethod
	def removeDup(lst: list) -> tuple[list, set]:
		res = []
		seen = set()
		for el in lst:
			if el == 0:
				res.append(el)
				continue
			if el not in seen:
				seen.add(el)
				res.append(el)
		return (res, seen)

	@staticmethod
	def findDayInSchedule(id: int, schedule: list[int]) -> int:
		day = 1
		for id_ in schedule:
			if id_ == 0:
				day += 1
			if id_ == id:
				return day
		return -1

	@staticmethod
	def getDayIndexesInSchedule(
			day: int, schedule: list[int]
		) -> tuple[int, int]:
		day_count = 1; start = 0; end = len(schedule)
		for idx, id in enumerate(schedule):
			if id != 0:
				continue
			day_count += 1
			if day_count == day:
				start = idx + 1
			if day_count == day + 1:
				end = idx
				break
		return (start, end)

	@staticmethod
	def splitScheduleByDay(schedule: list[int]) -> list[list[int]]:
		buffer = []
		res = []
		for id in schedule:
			if id == 0:
				res.append(buffer)
				buffer = []
				continue
			buffer.append(id)
		res.append(buffer)
		return res

	def evaluateTruckDaySchedule(
			self, day_count, day_schedule: list[int], detail: bool
	) -> TruckEvalSheet:
		class Buffer(self.TruckEvalSheet):
			dist_check = 0
			capacity = 0
			no_truck = 1
			prev_pos = (0, 0)

		buffer = Buffer()
		truck_route = []
		day_route = []

		def add_truck_route(slots: tuple[int], dist: float):
			buffer.total_truck_dist += dist
			buffer.dist_check += dist
			if detail != True:
				return
			for slot in slots:
				truck_route.append(slot)

		def add_new_truck(slot: int, total: int, new: int):
			buffer.no_truck += 1
			buffer.total_truck_dist += total
			buffer.dist_check = new
			if detail != True:
				return
			if len(truck_route) > 0:
				day_route.append(truck_route)
			truck_route = []
			truck_route.append(slot)

		def end_day():
			buffer.total_truck_dist += self.getDist(buffer.prev_pos, (0, 0))
			buffer.total_truck += buffer.no_truck
			if buffer.max_truck < buffer.no_truck:
				buffer.max_truck = buffer.no_truck
			if detail != True:
				return
			if len(truck_route) > 0:
				day_route.append(truck_route)
				truck_route = []
			buffer.schedule = day_route

		for slot in day_schedule:
			req = self.requests[slot - 1]
			new_cap = self.getCap(slot)
			next = self.getLoc(req.location)
			next_dist = self.getDist(buffer.prev_pos, next)
			ret_next_dist = self.getDist(next, (0, 0))
			ret_prev_dist = self.getDist(buffer.prev_pos, (0, 0))
			if buffer.capacity + new_cap > self.truck.capacity:
				if buffer.dist_check + ret_prev_dist + ret_next_dist < self.truck.max_distance:
					add_truck_route((0, slot), ret_prev_dist + ret_next_dist)
				else:
					add_new_truck(slot, ret_prev_dist + ret_next_dist, ret_next_dist)
				buffer.capacity = new_cap
			else:
				if buffer.dist_check + next_dist + ret_next_dist < self.truck.max_distance:
					add_truck_route((slot,), next_dist)
				else:
					add_new_truck(slot, ret_prev_dist + ret_next_dist, ret_next_dist)
				buffer.capacity += new_cap
			if day_count < req.first:
				buffer.truck_violation += 1
				buffer.penalty += self.hard_penalty * (req.first - day_count)
			elif day_count > req.last:
				buffer.truck_violation += 1
				buffer.penalty += self.hard_penalty * (day_count - req.last)
			buffer.prev_pos = next

		end_day()

		buffer.__class__ = self.TruckEvalSheet
		return buffer

	def evaluateTruckSchedule(
			self, schedule: list[int], detail: bool = False
		) -> TruckEvalSheet:
		eval_sheet = self.TruckEvalSheet()
		day_schedule = self.splitScheduleByDay(schedule)
		for idx, day in enumerate(day_schedule):
			if len(day) == 0:
				continue
			buffer = self.evaluateTruckDaySchedule(idx + 1, day, detail)
			eval_sheet.add(buffer)
		eval_sheet.cal_cost(self)
		return eval_sheet

	def truckScheduleInit(self) -> list[int]:
		added = []
		avaliable = []
		urgent = []
		res = []

		def get_valid_requests(day: int):
			for req in self.requests:
				id = req.id
				if day > req.last or day < req.first:
					continue
				if id in added:
					continue
				if day == req.last:
					urgent.append(id)
				else:
					avaliable.append(id)

		for day in range(1, self.days):
			buffer = []
			get_valid_requests(day)
			if len(avaliable) == 0 and len(urgent) == 0:
				res.append(0)
				continue
			if len(avaliable) > 0:
				no = random.randint(0, len(avaliable))
				buffer = random.sample(avaliable, no)
			buffer.extend(random.sample(urgent, len(urgent)))
			res.extend(buffer)
			added.extend(buffer)
			avaliable.clear()
			urgent.clear()
			res.append(0)
		return res

	def mutateSwapDayOrder(
			self, individual: list[int], indpb: float
		) -> tuple[list[int]]:
		if random.random() > indpb:
			return (individual, )
		day = random.randint(0, self.days)
		start, end = self.getDayIndexesInSchedule(day, individual)
		if end == start:
			return (individual, )
		# print(f'debug mut swap2 b {start}, {end}, {individual[start:end]}')
		# print(f'debug individual {individual}')
		buffer = random.sample(individual[start:end], end - start)
		# print('debug mut swap2 af')
		individual[start:end] = buffer
		return (individual, )
	
	def mutateExchangeDeliverDay(
			self, individual: list[int], indpb: float
		) -> tuple[list[int]]:
		def find_request(id):
			day = 1
			index = 0
			for idx, slot in enumerate(individual):
				if slot == 0:
					day += 1
				if slot == id:
					index = idx
			return (day, index)

		def change_request_day(id, day, index):
			req: 'Factory.Request' = self.requests[id - 1]
			new_day = random.choice(range(req.first, req.last))
			if new_day == day and new_day - 1 >= req.first:
				new_day -= 1
			start, end = self.getDayIndexesInSchedule(new_day, individual)
			new_index = start + 1
			if start + 1 < end:
				new_index = random.choice(range(start + 1, end))
			if new_index > index:
				new_index, index = index, new_index
			element = individual.pop(index)
			individual.insert(new_index, element)

		if random.random() > indpb:
			return (individual, )
		no = self.skewed_sample(1, self.requests[-1].id)
		req = random.sample(range(1, self.requests[-1].id + 1), no)
		for id in req:
			day, index = find_request(id)
			change_request_day(id, day, index)
		return (individual, )

	def crossoverTruck(
			self, parent1: list[int], parent2: list[int]
		) -> tuple[list[int]]:
		def find_breakpoint_index(breakpoint: int):
			index1 = 0
			index2 = 0
			day_count1 = 1
			day_count2 = 1
			for idx, (elem1, elem2) in enumerate(zip(parent1, parent2)):
				if elem1 == 0:
					day_count1 += 1
					if day_count1 == breakpoint:
						index1 = idx
				if elem2 == 0:
					day_count2 += 1
					if day_count2 == breakpoint:
						index2 = idx
			return (index1, index2)

		def repair_offspring(offspring: list):
			def add_missing_slot(lst: list, diff: list, ref: list):
				for slot in diff:
					day = self.findDayInSchedule(slot, ref)
					start, end = self.getDayIndexesInSchedule(day, lst)
					index = start + 1
					if start + 1 < end:
						index = random.randint(index, end)
					lst.insert(index, slot)

			res, seen = self.removeDup(offspring)
			seen.add(0)
			diff1 = [id for id in parent1 if id not in seen]
			diff2 = [id for id in parent2 if id not in seen and id not in diff1]
			add_missing_slot(res, diff1, parent1)
			add_missing_slot(res, diff2, parent2)
			return res

		breakpoint = random.randint(1, self.days)
		index1, index2 = find_breakpoint_index(breakpoint)
		buffer1 = repair_offspring(parent1[:index1] + parent2[index2:])
		buffer2 = repair_offspring(parent2[:index2] + parent1[index1:])
		offspring1 = self.makeCreatorTemplate()
		offspring2 = self.makeCreatorTemplate()
		offspring1.extend(buffer1); offspring2.extend(buffer2)
		# print(f'debug {len(offspring1)}, {len(offspring2)}')
		return (offspring1, offspring2)

	def scheduleInit(self) -> Schedule:
		truck = self.truck_schedule_ref
		already_delivered = []
		res: list[list[int]] = []
		buffer: list['Factory.TechStat'] = []

		def reorganize_schedule(tech: list):
			tech_part = []
			for t in tech:
				buffer = self.splitScheduleByDay(t)
				buffer.pop()
				tech_part.append(buffer)

			tech_buffer = []
			for idx in range(len(tech_part[0])):
				buffer = []
				for individual in tech_part:
					buffer.append(individual[idx])
				tech_buffer.append(buffer)
			return tech_buffer

		def avaliable_tech(machine_id: int, pos: tuple[int, int]):
			res = []
			for tech in self.technicians:
				if self.checkTechSkill(tech.id, machine_id) == False:
					continue
				if self.getDist(self.getLoc(tech.location), pos) * 2 < tech.dist_limit:
					res.append(tech.id)
			return res

		def assign_nearest_tech(
				tech_stat: list['Factory.TechStat'], tech_ids: list[int], pos: tuple[int, int]
			) -> int:
			dist = sys.maxsize
			id_res = 0
			# valid_id = []
			for id in tech_ids:
				ref = tech_stat[id - 1]
				if ref.avaliable() == False:
					continue
				check = self.getDist(ref.pos, pos)
				ret_dist = self.getDist(pos, ref.home)
				if check < dist and ref.dist + check + ret_dist < ref.dist_limit:
					# valid_id.append((id, check))
					id_res = id
					dist = check
				# if check + ret_dist < ref.dist_limit:
					# valid_id.append(id)
			# can append all possible id and random choice to find
			if id_res == 0:
				return -1

			# if len(valid_id) == 0:
				# return -1
			# id, dist = random.choice(valid_id)
			id = id_res
			ref = tech_stat[id - 1]
			ref.dist += dist
			ref.worked = True
			ref.pos = pos
			ref.req_count += 1
			return id

		for tech in self.technicians:
			res.append([])
			buffer.append(self.TechStat(
				self.getLoc(tech.location),
				tech.dist_limit, tech.req_limit, 5, 2
			))

		for delivered_today in truck:
			remain = []
			for id in already_delivered:
				req: 'Factory.Request' = self.requests[id - 1]
				loc = self.getLoc(req.location)
				tech = avaliable_tech(req.machine_id, loc)
				tech_id = assign_nearest_tech(buffer, tech, loc)
				if tech_id > 0:
					res[tech_id - 1].append(id)
				else:
					remain.append(id)
			already_delivered = remain
			already_delivered.extend(delivered_today)
			for tech in res:
				tech.append(0)
			for tech in buffer:
				tech.reset()
		tech_buffer : list[list[list[int]]] = reorganize_schedule(res)
		return self.Schedule(truck, tech_buffer)

	def evaluateTechDay(
			self, already_delivered: list[int],
			tech_stat: list[TechStat], tech_day: list[list[int]]
		) -> tuple[TechEvalSheet, list[int]]:
		res = self.TechEvalSheet(len(self.technicians))
		check = []

		def assign_penalty(idx: int, mul: int = 1):
			res.tech_violation[idx] += 1
			penalty = self.hard_penalty * mul
			res.tech_penalty[idx] += penalty
			res.total_tech_penalty += penalty

		# print(f'debug {tech_day}')
		for idx, tech in enumerate(tech_day):
			# print(f'debug {tech}')
			if len(tech) == 0:
				continue
			# print(f'debug {idx}, {len(tech_stat)} {len(tech_day)}')
			stat = tech_stat[idx]
			stat.worked = True
			res.resigterTech(idx)
			res.total_tech_deployed += 1
			for req in tech:
				# print(f'req check {req}')
				if req not in already_delivered or \
					self.checkTechSkill(idx + 1, self.requests[req - 1].machine_id) == False:
					# print(f'skill check')
					assign_penalty(idx)
					continue
				check.append(req)
				# print(f'debug {req}')
				loc = self.getLoc(self.requests[req - 1].location)
				stat.dist += self.getDist(loc, stat.pos)
				stat.pos = loc
				stat.req_count += 1

		for idx, tech in enumerate(tech_stat):
			tech.dist += self.getDist(tech.pos, tech.home)
			res.total_tech_dist += tech.dist
			if tech.full_work_shift == True and tech.worked:
				assign_penalty(idx)
			if tech.dist > tech.dist_limit:
				# print(f'dist limit {idx}')
				assign_penalty(idx, tech.dist - tech.dist_limit)
			if tech.req_count > tech.req_limit:
				assign_penalty(idx, tech.req_count - tech.req_limit)
			tech.reset()
			if tech.deploy_count > tech.deploy_limit:
				assign_penalty(idx)
			res.total_tech_penalty += tech.dist

		remain = list(set(already_delivered) - set(check))
		for id in remain:
			id = self.requests[id - 1].machine_id
			res.idle_machine_cost += self.machines[id - 1].penalty

		return (res, remain)

	def evaluateSchedule(
			self, schedule: Schedule, detail: bool = False
		) -> EvalSheet:
		eval_truck = self.TruckEvalSheet()
		eval_tech = self.TechEvalSheet(len(self.technicians))
		tech_status = [
			self.TechStat(
				self.getLoc(tech.location),
				tech.dist_limit, tech.req_limit, 5, 2
			)
			for tech in self.technicians
		]
		already_delivered = []

		for day_count, day_schedule in enumerate(schedule):
			truck_day, tech_day = tuple(day_schedule)
			# print(f'{type(truck_day)}, {type(tech_day)}, {type(tech_day[0])}, {type(tech_day[0][0])}')
			truck_score = self.evaluateTruckDaySchedule(day_count + 1, truck_day, detail)
			eval_truck.add(truck_score)
			tech_res = self.evaluateTechDay(already_delivered, tech_status, tech_day)
			eval_tech.add(tech_res[0])
			already_delivered = tech_res[1]
			already_delivered.extend(truck_day)

		res = self.EvalSheet(truck=eval_truck, tech=eval_tech)
		res.cal_cost(self)
		return res

	def mutateTechSwap(
			self, schedule: Schedule, indpb: float
		) -> tuple[Schedule]:
		
		def find_request(id: int) -> tuple[int, int, int]:
			for day_idx, day in enumerate(schedule):
				_, tech_day = tuple(day)
				for tech_idx, tech in enumerate(tech_day):
					for idx, id_ in enumerate(tech):
						if id == id_:
							return (day_idx + 1, tech_idx, idx)
			return (0, 0, 0)

		def find_avaliable_tech(id: int, tech: int) -> list[int]:
			buffer = []
			for tech_id in range(1, self.technicians[-1].id):
				if self.checkTechSkill(
					tech_id, self.requests[id - 1].machine_id
				) == True:
					if tech == tech_id:
						continue
					buffer.append(tech_id)
			return buffer

		if random.random() > indpb:
			return (schedule, )
		no = self.skewed_sample(1, self.requests[-1].id)
		req = random.sample(range(1, self.requests[-1].id + 1), no)
		for id in req:
			day, tech_idx, idx = find_request(id)
			if day == 0:
				continue
			buffer = find_avaliable_tech(id, tech_idx + 1)
			if len(buffer) == 0:
				continue
			new_tech_id = random.choice(buffer)
			_, tech = tuple(schedule[day - 1])
			new_idx = random.randint(0, len(tech[new_tech_id - 1]))
			tech[new_tech_id - 1].insert(new_idx, id)
			tech[tech_idx].pop(idx)
		return (schedule, )

	def mutateTechShift(self, schedule: Schedule, indpb: float) -> tuple[Schedule]:
		if random.random() > indpb:
			return (schedule,)
		day = random.randint(1, self.days - 3)
		tech_id = random.randint(1, self.technicians[-1].id)
		_, tech1 = tuple(schedule[day - 1])
		_, tech2 = tuple(schedule[day])
		tech2[tech_id - 1].extend(tech1[tech_id - 1])
		tech1[tech_id - 1].clear()
		return (schedule, )

	def mutateTechScramble(self, schedule: Schedule, indpb: float) -> tuple[Schedule]:
		if random.random() > indpb:
			return (schedule, )
		day = random.randint(1, self.days - 2)
		tech_id = random.randint(1, self.technicians[-1].id)
		_, tech = tuple(schedule[day - 1])
		random.shuffle(tech[tech_id - 1])
		return (schedule, )

	def crossoverTech(
			self, parent1: Schedule, parent2: Schedule
		) -> tuple[Schedule, Schedule]:
		offspring1 = deepcopy(parent1); offspring2 = deepcopy(parent2)

		def remove_dup(lst: list[int], seen: set[int]) -> list[int]:
			res = []
			for ele in lst:
				if ele not in seen:
					res.append(ele)
					seen.add(ele)
			return res

		def wrapper_remove_dup(
				para: tuple[int, list[int], list[int]], sets: tuple[set[int], set[int]]
			) -> tuple[list[int], list[int]]:
			lst1 = remove_dup(para[1], sets[0])
			lst2 = remove_dup(para[2], sets[1])
			return (lst1, lst2)

		def wrapper_swap_person(
				para: tuple[int, list[int], list[int]], placeholder = None
			) -> tuple[list[int], list[int]]:
			if para[0] % 2 == 0:
				return (para[1], para[2])
			return (para[2], para[1])

		def loop_schedule(func, para):
			for day, day2 in zip(offspring1, offspring2):
				_, tech1 = tuple(day)
				_, tech2 = tuple(day2)
				for idx, (person1, person2) in enumerate(zip(tech1, tech2)):
					person1, person2 = func((idx, person1, person2), para)

		def find_id(schedule: 'Factory.Schedule', id: int):
			for day_idx, day in enumerate(schedule):
				_, tech_day = tuple(day)
				for tech_idx, tech in enumerate(tech_day):
					if id in tech:
						return (day_idx, tech_idx)
			return (-1, -1)

		def repair_schedule(
				diff: set[int],
				offspring: 'Factory.Schedule', parent: 'Factory.Schedule'
			):
			for id in diff:
				day_idx, tech_idx = find_id(parent, id)
				target_tech = offspring[day_idx].technicians[tech_idx]
				if len(target_tech) == 0:
					target_tech.append(id)
					continue
				insert_pos = random.randint(0, len(target_tech) - 1)
				target_tech.insert(insert_pos, id)

		visited1 = set(); visited2= set()
		loop_schedule(wrapper_swap_person, None)
		loop_schedule(wrapper_remove_dup, (visited1, visited2))
		all_set = set(range(1, self.requests[-1].id + 1))
		diff1 = all_set - visited1; diff2 = all_set - visited2
		repair_schedule(diff1, offspring1, parent1)
		repair_schedule(diff2, offspring2, parent2)
		return (offspring1, offspring2)
		

def debug_split_on_zero(lst):
	result = []
	sublist = []
	
	for element in lst:
		if element == 0:
			result.append(sublist)
			sublist = []
		else:
			sublist.append(element)
	
	result.append(sublist)
	return result

def count_zero(lst: list[int]) -> int:
	count = 0
	for id in lst:
		if id == 0:
			count += 1
	return count

def test_len(factory, tech):
	test = 0
	for t in tech:
		buffer = factory.splitScheduleByDay(t)
		if test == 0:
			test = len(buffer)
		if test != len(buffer):
			print(f'error {test} {len(buffer)}')


def main():
	def empty_list():
		return []
	seed = random.randint(1, 999)
	seed = 165
	print(f'seed {seed}')
	factory = Factory('test_1.txt', seed=seed)
	factory.resigterIndividualClass(empty_list)
	truck = factory.truckScheduleInit()
	# print(truck)
	eval = factory.evaluateTruckSchedule(truck, False)
	print(eval.cal_cost(factory))
	factory.resigterTruckScheduleRef(truck)
	schedule1 = factory.scheduleInit()
	eval = factory.evaluateSchedule(schedule1)
	print(eval)
	# schedule2 = factory.scheduleInit()
	# t1, t2 = factory.crossoverTech(schedule1, schedule2)
	# eval1 = factory.evaluateSchedule(t1)
	# eval2 = factory.evaluateSchedule(t2)
	# print(eval1)
	# print(eval1.cal_cost(factory))
	# print(eval2.cal_cost(factory))

if __name__ == "__main__":
	main()
