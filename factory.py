import random
import math
from copy import deepcopy
from typing import Type, TypeVar, List, Dict, Tuple, NamedTuple

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
			hard_penalty = (max(truck.values()) + max(tech_cost.values())) * 100
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

	def truckScheduleInit(self) -> List[int]:
		buffer = range(self.requests[0].id, self.requests[-1].id + self.days - 1)
		buffer = random.sample(buffer, len(buffer))
		return [id if id < self.requests[-1].id else 0 for id in buffer]

	def evaluateTruckScedule(
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
			# print(f"debug no_truck {buffer.no_truck}")
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

def mutate_swap(genes: List[int], indpb: float) -> Tuple[List[int]]:
	"""
	Mutate a list of genes by swapping two random positions in the list.
	
	Parameters:
	genes: List[int]  - The gene list to mutate (mutated in place).
	indpb: float      - The probability of mutation. If the random value
						is less than indpb, the swap will occur.
	Returns: Tuple[List[int]] - A tuple containing the mutated gene list
	"""
	if random.random() > indpb:
		return (genes,)
	i, j = random.sample(range(len(genes)), 2)
	genes[i], genes[j] = genes[j], genes[i]
	return (genes, )

def mutate_inversion(genes: List[int], indpb: float) -> Tuple[List[int]]:
	"""
	Apply inversion mutation to a list of genes with a given probability.

	This function randomly selects a section of genes and reverses the order 
	of the genes in that section. The rest of the genes remain unchanged.

	Parameters:
	genes: List[int]  
		The individual to be mutated, represented as a list of integers (genes).
		
	indpb: float
		The probability that a mutation (inversion) will occur. If a random value 
		is less than `indpb`, the mutation will take place.

	Returns:
	Tuple[List[int]] 
		A tuple containing the mutated gene list.
	"""
	if random.random() > indpb:
		return (genes, )
	start, end = sorted(random.sample(range(len(genes)), 2))
	genes[start:end+1] = reversed(genes[start:end+1])
	return (genes,)

def mutate_scramble(genes: List[int], indpb:float) -> Tuple[List[int]]:
	"""
	Apply scramble mutation to a list of genes with a given probability.

	This function randomly selects a section of genes and randomly scrambles 
	the order of the genes in that section. The rest of the genes remain unchanged.

	Parameters:
	genes: List[int]  
		The individual to be mutated, represented as a list of integers (genes).
		
	indpb: float
		The probability that a mutation (scramble) will occur. If a random value 
		is less than `indpb`, the mutation will take place.

	Returns:
	Tuple[List[int]] 
		A tuple containing the mutated gene list.
	"""
	if random.random() > indpb:
		return (genes, )
	start, end = sorted(random.sample(range(len(genes)), 2))
	section = genes[start:end+1]
	random.shuffle(section)
	genes[start:end+1] = section
	return (genes,)

def alternating_edge_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int]]:
	"""
	Perform the Alternating Edge Crossover (AX) on two parent permutations.

	Parameters:
	parent1: List[int]
		First parent permutation (list of integers).
		
	parent2: List[int]
		Second parent permutation (list of integers).

	Returns:
	Tuple[List[int], List[int]]
		A tuple containing two offspring lists resulting from the crossover.
	"""
	# Initialize offspring with empty values
	offspring1 = deepcopy(parent1)
	offspring2 = deepcopy(parent2)
	
	# Randomly select a starting point
	start = random.randint(0, len(parent1) - 1)

	# Set the first element of both offspring as the element from parent1 and parent2 respectively
	offspring1[start] = parent1[start]
	offspring2[start] = parent2[start]

	# The current parent alternates between parent1 and parent2
	current_parent = 2  # Start by selecting from parent2 after the initial element from parent1
	for i in range(1, len(parent1)):
		if current_parent == 2:
			# Select element from parent2
			value = parent2[(start + i) % len(parent2)]
			# If it's already in offspring1, select from parent1
			if value not in offspring1:
				offspring1[(start + i) % len(parent1)] = value
			else:
				value = parent1[(start + i) % len(parent1)]
				offspring1[(start + i) % len(parent1)] = value
			# Select element from parent1 for offspring2
			value = parent1[(start + i) % len(parent1)]
			if value not in offspring2:
				offspring2[(start + i) % len(parent1)] = value
			else:
				value = parent2[(start + i) % len(parent2)]
				offspring2[(start + i) % len(parent1)] = value
			current_parent = 1
		else:
			# Select element from parent1 for offspring1
			value = parent1[(start + i) % len(parent1)]
			if value not in offspring1:
				offspring1[(start + i) % len(parent1)] = value
			else:
				value = parent2[(start + i) % len(parent2)]
				offspring1[(start + i) % len(parent1)] = value
			# Select element from parent2 for offspring2
			value = parent2[(start + i) % len(parent2)]
			if value not in offspring2:
				offspring2[(start + i) % len(parent1)] = value
			else:
				value = parent1[(start + i) % len(parent1)]
				offspring2[(start + i) % len(parent1)] = value
			current_parent = 2
	
	return (offspring1, offspring2)

def ordered_crossover(parent1: list, parent2: list) -> Tuple[list, list]:
	"""Performs ordered crossover (OX1) on two parents."""
	
	size = len(parent1)
	
	# Step 1: Select random crossover points
	start, end = sorted(random.sample(range(size), 2))
	
	# Step 2: Create offspring initialized with empty values (-1 as placeholder)
	offspring1 = deepcopy(parent1)
	offspring2 = deepcopy(parent2)
	
	# Step 3: Copy the segment from each parent
	offspring1[start:end+1] = parent1[start:end+1]
	offspring2[start:end+1] = parent2[start:end+1]
	
	# Step 4: Fill remaining positions from the other parent in order
	def fill_offspring(offspring, parent):
		pos = (end + 1) % size  # Start filling from the next position after the copied segment
		for gene in parent:
			if gene not in offspring:  # Only add if it's not already in the offspring
				offspring[pos] = gene
				pos = (pos + 1) % size  # Move to the next position
	
	fill_offspring(offspring1, parent2)
	fill_offspring(offspring2, parent1)
	
	return offspring1, offspring2

def main():
	factory = Factory('test_1.txt', seed=200)
	# print(factory)
	t1 = factory.truckScheduleInit()
	print(t1)
	summary = {}
	detail = []
	cost = factory.evaluateTruckScedule(t1, summary, detail)
	print(cost)
	print(summary)
	print(f'debug {len(detail)}')
	for index, day in enumerate(detail):
		print(f"day {index + 1}")
		for truck in day:
			print(truck)
	p1 = factory.getLoc(43); p2 = factory.getLoc(5); p3 = (0, 0)
	dist1 = factory.getDist(p1, p3); dist2 = factory.getDist(p2, p3)
	dist3 = factory.getDist(p1, p2)
	cap1 = factory.getCap(43); cap2 = factory.getCap(5)
	print(f'max dist {factory.truck.max_distance}, max cap {factory.truck.capacity}')
	print(f'debug, loc 1{p1} dist {dist1}')
	print(f'debug, loc 2{p2} dist {dist2}')
	print(f'dist between {dist3}, total {dist1 + dist3 + dist2}, v2 {dist1 * 2 + dist2 * 2}')
	print(f'capacity {cap1} {cap2}, total {cap1 + cap2}')

if __name__ == "__main__":
	main()