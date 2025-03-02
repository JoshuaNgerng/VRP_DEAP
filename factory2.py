import random
import math
import sys
import numpy as np
from copy import deepcopy
from deap import base
from typing import TypeVar, List, Dict, Tuple, NamedTuple, Protocol, Callable

class IndividualTemplate(Protocol):
	fitness: base.Fitness

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
	class TruckEvalSheet(NamedTuple):
		total_dist: float
		max_truck: int
		total_truck: int
		violation: int
		schedule: dict[int, list[list[int]]]

	days: int
	truck: Truck
	tech_cost: TechCost
	machines: List[Machine]
	locations: List[Location]
	requests: List[Request]
	technicians: List[Technician]
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
		def sort(src: List[T]) -> List[T]:
			return sorted(src, key=lambda src: src.id)

		def parse_map(
				dst: List[T], ref: List[T],
				start: int, no: int, lines: List[str]
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
				dst: List[T], ref: List[T],
				start:int, no: int, lines: List[str]
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
				# print(f"line: {line}")
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

	def getLoc(self, id: int) -> Tuple[int]:
		return (self.locations[id - 1].x, self.locations[id - 1].y)

	def getCap(self, id: int) -> int:
		req = self.requests[id - 1]
		size = self.machines[req.machine_id - 1].weight
		return size * req.machine_quantity

	def getDist(self, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
		x1, y1 = pt1; x2, y2 = pt2
		return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

	def checkTechSkill(self, tech_id: int, machine_id: int) -> bool:
		bits = self.technicians[tech_id - 1].machine
		return (bits & (1 << machine_id - 1)) != 0

	# def evaluateTruckDaySchedule(
	# 		self, day_schedule: list[int], detail_schedule: bool
	# ) -> TruckEvalSheet:

	def evaluateTruckSchedule(
			self, schedule: List[int],
			detail_summary: Dict[str, int] | None = None,
			detail_schedule: List[List[int]] | None = None
		) -> int:
		class Buffer:
			day_count = 1
			total_cost = 0
			total_dist = 0
			dist = 0
			capacity = 0
			no_truck = 0
			total_truck = 0
			max_truck = 0
			violation = 0
			prev_pos = (0, 0)

		buffer = Buffer()
		truck_route = []
		day_route = []
		detail = detail_schedule is not None

		def add_truck_route(slots: Tuple[int], dist: int):
			buffer.total_dist += dist
			buffer.dist += dist
			if detail is True:
				for slot in slots:
					truck_route.append(slot)

		def add_new_truck(slot: int, total: int, new: int):
			buffer.no_truck += 1
			buffer.total_dist += total
			buffer.dist = new
			if detail is not True:
				return
			if len(truck_route) > 0:
				day_route.append(truck_route.copy())
			truck_route.clear()
			truck_route.append(slot)

		def end_day():
			buffer.day_count += 1
			buffer.total_dist += self.getDist(buffer.prev_pos, (0, 0))
			buffer.prev_pos = (0, 0)
			buffer.total_truck += buffer.no_truck
			if buffer.max_truck < buffer.no_truck:
				buffer.max_truck = buffer.no_truck
			buffer.no_truck = 0
			buffer.capacity = 0
			buffer.dist = 0
			if detail is not True:
				return
			if len(truck_route) > 0:
				day_route.append(truck_route.copy())
			detail_schedule.append(day_route.copy())
			truck_route.clear()
			day_route.clear()

		for slot in schedule:
			if slot == 0:
				end_day()
				continue
			if buffer.no_truck == 0:
				buffer.no_truck += 1
			req = self.requests[slot - 1]
			new_cap = self.getCap(slot)
			next = self.getLoc(req.location)
			next_dist = self.getDist(buffer.prev_pos, next)
			ret_next_dist = self.getDist(next, (0, 0))
			ret_prev_dist = self.getDist(buffer.prev_pos, (0, 0))
			if buffer.capacity + new_cap > self.truck.capacity:
				if buffer.dist + ret_prev_dist + ret_next_dist < self.truck.max_distance:
					add_truck_route((0, slot), ret_prev_dist + ret_next_dist)
				else:
					add_new_truck(slot, ret_prev_dist + ret_next_dist, ret_next_dist)
				buffer.capacity = new_cap
			else:
				if buffer.dist + next_dist + ret_next_dist < self.truck.max_distance:
					add_truck_route((slot,), next_dist)
				else:
					add_new_truck(slot, ret_prev_dist + ret_next_dist, ret_next_dist)
				buffer.capacity += new_cap
			if buffer.day_count < req.first:
				buffer.violation += 1
				buffer.total_cost += self.hard_penalty * (req.first - buffer.day_count)
			elif buffer.day_count > req.last:
				buffer.violation += 1
				buffer.total_cost += self.hard_penalty * (buffer.day_count - req.last)
			buffer.prev_pos = next

		if buffer.prev_pos != (0, 0):
			buffer.dist += self.getDist(buffer.prev_pos, (0, 0))
		if buffer.dist > 0:
			buffer.total_dist += buffer.dist
		if buffer.no_truck > 0:
			buffer.total_truck += buffer.no_truck
			if buffer.max_truck < buffer.no_truck:
				buffer.max_truck = buffer.no_truck
		buffer.total_cost += buffer.total_dist * self.truck.distance_cost + \
			buffer.total_truck * self.truck.day_cost + \
			buffer.max_truck * self.truck.cost
		buffer.total_cost = int(buffer.total_cost)

		if detail_summary is not None:
			detail_summary.clear()
			detail_summary['total_distance'] = buffer.total_dist
			detail_summary['total_no_truck'] = buffer.total_truck
			detail_summary['max_no_truck'] = buffer.max_truck
			detail_summary['violations'] = buffer.violation

		if detail_schedule is None:
			return buffer.total_cost

		if len(truck_route) > 0:
			day_route.append(truck_route.copy())
		if len(day_route) > 0:
			detail_schedule.append(day_route.copy())
		return buffer.total_cost

	def truckScheduleInit(self) -> List[int]:
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
			self, individual: IndividualTemplate, indpb: float
		) -> Tuple[IndividualTemplate]:
		if random.random() > indpb:
			return (individual, )
		day = random.randint(0, self.days)
		start = 0
		end = len(individual) - 1
		count = 1
		for idx, slot in enumerate(individual):
			if slot > 0:
				continue
			count += 1
			if count == day:
				start = idx
			if count == day + 1:
				end = idx
				break
		if end == start:
			return (individual, )
		buffer = random.sample(individual[start:end], end - start)
		individual[start:end] = buffer
		return (individual, )
	
	def mutateExchangeDeliverDay(
			self, individual: IndividualTemplate, indpb: float
		) -> Tuple[IndividualTemplate]:
		def skewed_sample(start, end, scale=5):
			value = int(np.random.exponential(scale)) + start
			return min(value, end)

		def find_request(id, individual):
			day = 1
			index = 0
			for idx, slot in enumerate(individual):
				if slot == 0:
					day += 1
				if slot == id:
					index = idx
			return (day, index)

		def change_request_day(id, day, index, individual):
			req = self.requests[id - 1]
			new_day = random.choice(range(req.first, req.last))
			if new_day == day and new_day - 1 >= req.first:
				new_day -= 1
			count = 1
			start = 0
			end = len(individual) - 1
			for idx, slot in enumerate(individual):
				if slot > 0:
					continue
				count += 1
				if count == new_day:
					start = idx
				if count == new_day + 1:
					end = idx
					break
			new_index = start + 1
			if start + 1 < end:
				new_index = random.choice(range(start + 1, end))
			if new_index > index:
				new_index, index = index, new_index
			element = individual.pop(index)
			individual.insert(new_index, element)

		if random.random() > indpb:
			return (individual, )
		no = skewed_sample(1, self.requests[-1].id)
		req = random.sample(range(1, self.requests[-1].id + 1), no)
		for id in req:
			day, index = find_request(id, individual)
			change_request_day(id, day, index, individual)
		return (individual, )

	def crossoverTruck(
			self, parent1: IndividualTemplate, parent2: IndividualTemplate
		) -> Tuple[IndividualTemplate]:
		def find_breakpoint_index(breakpt):
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

		def repair_offspring(offspring):
			def remove_duplicates(lst):
				seen = set()  # Keep track of seen elements
				result = []   # List to store the result without duplicates
				for num in lst:
					if num == 0:
						result.append(num)
						continue
					if num not in seen:
						result.append(num)
						seen.add(num)
				return result

			def find_day(lst, id):
				day_count = 1
				for id_ in lst:
					if id_ == 0:
						day_count += 1
						continue
					if id_ == id:
						return day_count
				return day_count

			def find_day_range(lst, day):
				start = 0
				end = len(lst) - 1
				day_count = 1
				for idx, id in enumerate(lst):
					if id > 0:
						continue
					day_count += 1
					if day_count == day:
						start = idx
					if day_count - 1 == day:
						end = idx
						break
				return (start, end)

			def add_missing_slot(lst, diff, ref):
				for slot in diff:
					day = find_day(ref, slot)
					start, end = find_day_range(lst, day)
					index = start + 1
					if start + 1 < end:
						index = random.randint(index, end)
					lst.insert(index, slot)

			buffer = remove_duplicates(offspring)
			diff1 = [id for id in parent1 if id not in buffer]
			diff2 = [id for id in parent2 if id not in buffer and id not in diff1]
			add_missing_slot(buffer, diff1, parent1)
			add_missing_slot(buffer, diff2, parent2)
			offspring.clear()
			offspring.extend(buffer)
			return offspring

		breakpoint = random.randint(1, self.days)
		index1, index2 = find_breakpoint_index(breakpoint)
		offspring1 = parent1[:index1]; offspring1b = parent1[index1:]
		offspring2 = parent2[:index2]; offspring2b = parent2[index2:]
		buffer1 = repair_offspring(offspring1 + offspring2b)
		buffer2 = repair_offspring(offspring2 + offspring1b)
		offspring1 = deepcopy(parent1); offspring2 = deepcopy(parent2)
		offspring1.clear(); offspring2.clear()
		offspring1.extend(buffer1); offspring2.extend(buffer2)
		return (offspring1, offspring2)

	def technicianInit(self, truck: IndividualTemplate) -> List[List[int]]:
		class Buffer:
			dist: int = 0
			dist_limit = 0
			worked: bool = False
			deploy_count: int = 0
			deploy_limit: int = 0
			pos: Tuple[int, int] = (0, 0)
			home: Tuple[int, int] = (0, 0)
			def __init__(self, start: Tuple[int, int], dist_limit: int, deploy_limit):
				self.pos = start; self.home = start
				self.dist_limit = dist_limit; self.deploy_limit = deploy_limit
			def reset(self):
				self.dist = 0; self.pos = self.home
				if self.worked == True:
					self.deploy_count += 1
				else:
					self.deploy_count = 0
				self.worked = False

		visited = 0
		truck_index = 0
		deliver_today = []
		already_delivered = []
		res: list[list[int]] = []
		buffer: list[Buffer] = []

		def get_delivered_today():
			nonlocal truck_index
			deliver_today.clear()
			while truck_index < len(truck):
				if truck[truck_index] == 0:
					truck_index += 1
					return
				deliver_today.append(truck[truck_index])
				truck_index += 1

		def avaliable_tech(machine_id: int, pos: Tuple[int, int]):
			res = []
			for tech in self.technicians:
				if self.checkTechSkill(tech.id, machine_id) == False:
					continue
				if self.getDist(self.getLoc(tech.location), pos) * 2 < tech.dist_limit:
					res.append(tech.id)
			return res

		def assign_nearest_tech(
				buffer: list[Buffer], tech: list[int], pos: Tuple[int, int]
			) -> int:
			dist = sys.maxsize
			id_res = 0
			#valid_id = []
			for id in tech:
				ref = buffer[id - 1]
				if ref.deploy_count >= ref.deploy_limit:
					continue
				check = self.getDist(ref.pos, pos)
				ret_dist = self.getDist(pos, ref.home)
				if check < dist and check + ret_dist < ref.dist_limit:
					id_res = id
					dist = check
				# if check + ret_dist < ref.dist_limit:
					# valid_id.append(id)
			# can append all possible id and random choice to find
			# if len(valid_id) == 0:
			# return -1

			if id_res == 0:
				return -1
			ref = buffer[id_res - 1]
			ref.dist += dist
			ref.worked = True
			return id_res
			# return random.choice(valid_id)

		for tech in self.technicians:
			res.append([])
			buffer.append(Buffer(
				self.getLoc(tech.location)), tech.dist_limit, tech.req_limit
			)

		while visited != len(self.requests):
			get_delivered_today()
			remain = []
			for id in already_delivered[:]:
				req = self.requests[id - 1]
				loc = self.getLoc(req.location)
				tech = avaliable_tech(req.machine_id, loc)
				tech_id = assign_nearest_tech(buffer, tech, loc)
				if tech_id > 0:
					res[tech_id - 1].append(id)
					visited += 1
				else:
					remain.append(id)
			already_delivered = remain
			already_delivered.extend(deliver_today)
			for tech in res:
				tech.append(0)
			for slot in buffer:
				slot.reset()
		return res

	def evaluateSchedule(
			self, truck: list[int], tech: list[list[int]],
			detail_summary: dict[str, int] | None = None,
			detail_schedule: list[list[int]] | None = None
		) -> int:
		class BufferTruck:
			total_dist = 0
			no_truck = 0
			total_truck = 0
			max_truck = 0
			violation = 0

		class BufferIndividualTech:
			dist: int = 0
			dist_limit = 0
			worked: bool = False
			deploy_count: int = 0
			deploy_limit: int = 0
			pos: Tuple[int, int] = (0, 0)
			home: Tuple[int, int] = (0, 0)
			def __init__(self, start: Tuple[int, int], dist_limit: int, deploy_limit):
				self.pos = start; self.home = start
				self.dist_limit = dist_limit; self.deploy_limit = deploy_limit
			def reset(self):
				self.dist = 0; self.pos = self.home
				if self.worked == True:
					self.deploy_count += 1
				else:
					self.deploy_count = 0
				self.worked = False
		class BufferTech:
			total_dist = 0
			no_tech = 0
			total_tech = 0
			tech_deployed = 0
			violation = 0
			tech = []
			def __init__(self, factory: Factory):
				self.tech = [
					BufferIndividualTech(
						factory.getLoc(tech.location),
						tech.dist_limit, tech.req_limit
					)
					for tech in factory.technicians
				]
			def resigterTech(self, id: int):
				bitmask = 1 << (id - 1)
				self.tech_deployed |= bitmask
			def checkTechStatus(self, id: int) -> bool:
				bitmask = 1 << (id - 1)
				return (self.tech_deployed & bitmask) != 0
			def totalTechResigter(self) -> int:
				return bin(self.tech_deployed).count('1')

		def split_schedule(
				truck: list[int], tech: list[list[int]]
			) -> list[Tuple[list[int], list[list[int]]]]:
			def split_list(lst: list[int]) -> list[list[int]]:
				buffer = []
				res = []
				for id in truck:
					if id == 0:
						res.append(buffer)
						buffer = []
						continue
					buffer.append(id)
				return res
			truck_part = split_list(truck)
			tech_buffer = []
			for t in tech:
				tech_buffer.append(split_list(t))
			res = []
			for j in range(tech_buffer):
				buffer = []
				for i in range(len(tech_buffer[0])):
					buffer.append(tech_buffer[j][i])
				res.append((truck_part[j], buffer))
			return res

		day_count = 1
		total_cost = 0
		buffer_truck = BufferTruck()
		buffer_tech = BufferTech(self)
		already_delivered = []
		day_schedule = split_schedule(truck, tech)
		detail_day_schedule = ([], [])

		def evaluate_truck(day_count, truck_day, detail_schedule):
			no_truck = 1
			capacity = 0
			prev_pos = (0, 0)
			for id in truck_day:
				if id == 0:
					pass


		def evalute_tech(day_count, tech_day, already_delivered, detail_schedule):
			pass


		for truck_day, tech_day in day_schedule:
			evaluate_truck(day_count, truck_day, detail_day_schedule)
			# after process
			already_delivered.extend(truck_day)
			day_count += 1

		if detail_summary is not None:
			pass
		if detail_schedule is None:
			return total_cost
		return total_cost

def debug_split_on_zero(lst):
	# Initialize an empty list to store sublists
	result = []
	sublist = []
	
	for element in lst:
		if element == 0:
			# If we encounter a 0, save the current sublist and reset it
			result.append(sublist)
			sublist = []
		else:
			# Otherwise, add the element to the current sublist
			sublist.append(element)
	
	# Append the last sublist if it's not empty
	result.append(sublist)
	
	return result

def main():
	factory = Factory('test_1.txt', seed=200)
	print(factory.requests[1 - 1])
	t1 = factory.truckScheduleInit()
	t2 = factory.truckScheduleInit()
	summary = {}
	detail = []
	cost = factory.evaluateTruckScedule(t1, summary, detail)
	print(cost)
	print(summary)
	# for index, day in enumerate(detail):
		# print(f"day {index + 1}")
		# for truck in day:
			# print(truck)
	summary = {}
	off1, off2 = factory.crossoverDay(t1, t2)
	print(f'debug crossover {factory.evaluateTruckScedule(off1, detail_summary=summary)}')
	print(summary)
	summary = {}
	print(f'debug crossover {factory.evaluateTruckScedule(off2, detail_summary=summary)}')
	print(summary)
	print(f'debug len {len(t1)}, {len(t2)}, {len(off1)}, {len(off2)}')
	# p1 = factory.getLoc(43); p2 = factory.getLoc(5); p3 = (0, 0)
	# dist1 = factory.getDist(p1, p3); dist2 = factory.getDist(p2, p3)
	# dist3 = factory.getDist(p1, p2)
	# cap1 = factory.getCap(43); cap2 = factory.getCap(5)
	# print(f'max dist {factory.truck.max_distance}, max cap {factory.truck.capacity}')
	# print(f'debug, loc 1{p1} dist {dist1}')
	# print(f'debug, loc 2{p2} dist {dist2}')
	# print(f'dist between {dist3}, total {dist1 + dist3 + dist2}, v2 {dist1 * 2 + dist2 * 2}')
	# print(f'capacity {cap1} {cap2}, total {cap1 + cap2}')

if __name__ == "__main__":
	main()