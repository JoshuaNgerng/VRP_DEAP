import random
import math
from typing import List, Dict, Tuple

class Factory:
	day = 0
	truck = {
		"truck_capacity": 0, "truck_max_distance": 0,
		"truck_distance_cost": 0, "truck_day_cost": 0, "truck_cost": 0
	}
	tech_cost = {
		"technician_distance_cost": 0,
		"technician_day_cost": 0, "technician_cost": 0
	}
	machines : List[Dict[str, int]] = []
	location : List[Dict[str, int]] = []
	requests : List[Dict[str, int]] = []
	technicians : List[Dict[str, int]] = []
	hard_penalty = 10000

	def __init__(
			self, fname: str,
			hard_penalty: int | None = None, seed: int | None = None
		):
		machine = [ "id", "weight", "penalty" ]
		location = ["id", "x", "y"]
		request = [
			"id", "location", "first", "last",
			"machine_id", "machine_quant"
		]
		technician = [
			"id", "location",
			"dist_limit", "req_limit", "machine"
		]
		mapping = {
			"machines": (self.machines, machine),
			"locations": (self.location, location),
			"requests": (self.requests, request)
		}
		def parse_map(
				dst: List[Dict[str, int]], ref: List[str],
				start: int, no: int, lines: List[str]
			):
			"""
			Generic function that parses a space-separated line
			into a dictionary based on the provided template.
			"""
			count = 0
			for line in lines[start:]:
				if count == no:
					break
				# print(f"par masing {line}")
				values = list(map(int, line.split()))
				dst.append(dict(zip(ref, values)))
				count += 1

		def parse_tech(
				dst: List[Dict[str, int]], ref: List[str],
				start:int, end: int, lines: List[str]
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
				dst.append(dict(zip(ref, values)))
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
					self.day = int(no)
				elif key in self.truck:
					self.truck[key] = no
				elif key in self.tech_cost:
					self.tech_cost[key] = no
				elif key in mapping:
					dst, ref = mapping[key]
					parse_map(dst, ref, i, no, lines)
					i += no
				else:
					parse_tech(self.technicians, technician, i, no, lines)
					i += no
		if hard_penalty is None:
			hard_penalty = (max(self.truck.values()) + max(self.tech_cost.values())) * 100
		self.hard_penalty = hard_penalty
		if isinstance(seed, int):
			random.seed(seed)

	def __str__(self) -> str:
		s = f"DAYS: {self.day}\n"
		s += str(self.truck) + '\n'
		s += str(self.tech_cost) + '\n'
		s += "machines\n"
		for mach in self.machines:
			s += str(mach) + '\n'
		s += "locations\n"
		for loc in self.location:
			s += str(loc) + '\n'
		s += "requests\n"
		for req in self.requests:
			s += str(req) + '\n'
		s += "technician\n"
		for tech in self.technicians:
			s += str(tech) + '\n'
		return s

	def getLoc(self, id: int) -> Tuple[int]:
		x = self.location[id - 1]['x']
		y = self.location[id - 1]['y']
		return (x, y)

	def getCap(self, id: int) -> int:
		req = self.requests[id - 1]
		size = self.machines[req['machine_id'] - 1]['weight']
		return size * req['machine_quant']

	def getDist(self, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
		x1, y1 = pt1
		x2, y2 = pt2
		return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

	def truckScheduleInit(self) -> List[int]:
		buffer = range(self.requests[0]["id"], self.requests[-1]["id"] + self.day)
		buffer = random.sample(buffer, len(buffer))
		return [id if id < self.requests[-1]["id"] else 0 for id in buffer]

	def truckCost(self, schedule: List[int]) -> int:
		total_cost = 0
		day = 1
		total_dist = 0
		dist = 0
		capacity = 0
		no_truck = 0
		total_truck = 0
		max_truck = 0
		prev = (0, 0)

		for slot in schedule:
			if slot == 0:
				day += 1
				total_dist += self.getDist(prev, (0, 0))
				prev = (0, 0)
				total_truck += no_truck
				if max_truck < no_truck:
					max_truck = no_truck
				no_truck = 0
				capacity = 0
				continue
			if no_truck == 0:
				no_truck = 1
			req = self.requests[slot - 1]
			new_cap = self.getCap(slot)
			next = self.getLoc(req['location'])
			next_dist = self.getDist(prev, next)
			ret_next_dist = self.getDist(next, (0, 0))
			ret_prev_dist = self.getDist(prev, (0, 0))
			if capacity + new_cap > self.truck['truck_capacity']:
				if dist + ret_prev_dist + ret_next_dist < self.truck['truck_max_distance']:
					dist += ret_prev_dist + ret_next_dist
					total_dist += ret_prev_dist + ret_next_dist
				else:
					no_truck += 1
					total_dist += ret_prev_dist + ret_next_dist
					dist = ret_next_dist
				capacity = new_cap
			else:
				if dist + next_dist + ret_next_dist < self.truck['truck_max_distance']:
					dist += next_dist
					total_dist += next_dist
				else:
					no_truck += 1
					total_dist += ret_prev_dist + ret_next_dist
					dist = ret_next_dist
				capacity += new_cap
			if day < req['first']:
				total_cost += (req['first'] - day) * self.hard_penalty
			elif day > req['last']:
				total_cost += (req['last'] - day) * self.hard_penalty
			prev = next
		if prev != (0, 0):
			total_dist += dist
		if no_truck > 0:
			total_truck += no_truck
			if max_truck < no_truck:
				max_truck = no_truck
		total_cost += total_dist * self.truck['truck_distance_cost'] + \
			total_truck * self.truck['truck_day_cost'] + max_truck * self.truck['truck_cost']
		return total_cost

	def sumTruckSchedule(
			self, schedule: List[int]
		) -> Tuple[Dict[str, int], List[List[List[int]]]]:
		truck_info = {
			'total_dist': 0, 'no_truck_day': 0, 'no_truck_used': 0, 'violation': 0
		}
		truck_route = []
		day_schedule = []
		details = []
		day = 0
		dist = 0
		capacity = 0
		no_truck = 0
		max_truck = 0
		prev = (0, 0)
		for slot in schedule:
			if slot == 0:
				day += 1
				truck_info['total_dist'] += self.getDist(prev, (0, 0))
				prev = (0, 0)
				truck_info['no_truck_day'] += no_truck
				if max_truck < no_truck:
					max_truck = no_truck
				# print(no_truck)
				no_truck = 0
				capacity = 0
				if len(truck_route) > 0:
					day_schedule.append(truck_route.copy())
				details.append(day_schedule.copy())
				truck_route.clear()
				day_schedule.clear()
				dist = 0
				continue
			if no_truck == 0:
				no_truck = 1
			# print(f"debug {slot}")
			req = self.requests[slot - 1]
			new_cap = self.getCap(slot)
			next = self.getLoc(req['location'])
			next_dist = self.getDist(prev, next)
			ret_next_dist = self.getDist(next, (0, 0))
			ret_prev_dist = self.getDist(prev, (0, 0))
			if capacity + new_cap > self.truck['truck_capacity']:
				if dist + ret_prev_dist + ret_next_dist < self.truck['truck_max_distance']:
					dist += ret_prev_dist + ret_next_dist
					truck_info['total_dist'] += ret_prev_dist + ret_next_dist
					truck_route.append(0)
					truck_route.append(slot)
					# print(f"debug add slot return depo {slot}")
				else:
					no_truck += 1
					truck_info['total_dist'] += ret_prev_dist + ret_next_dist
					dist = ret_next_dist
					if len(truck_route) > 0:
						day_schedule.append(truck_route.copy())
					truck_route.clear()
					truck_route.append(slot)
					# print(f"debug new truck capacity {slot}")
				capacity = new_cap
			else:
				if dist + next_dist + ret_next_dist < self.truck['truck_max_distance']:
					dist += next_dist
					truck_info['total_dist'] += next_dist
					# print(f"debug add slot {slot}")
					truck_route.append(slot)
				else:
					no_truck += 1
					# print(f'huh dist {dist}')
					truck_info['total_dist'] += ret_prev_dist + ret_next_dist
					dist = ret_next_dist
					if len(truck_route) > 0:
						day_schedule.append(truck_route.copy())
					truck_route.clear()
					truck_route.append(slot)
					# print(f"debug add slot over max dist {slot}")
				capacity += new_cap
			if day < req['first']:
				truck_info['violation'] += 1
			elif day > req['last']:
				truck_info['violation'] += 1
			prev = next
		if prev != (0, 0):
			truck_info['total_dist'] += dist
		if no_truck > 0:
			truck_info['no_truck_day'] += no_truck
			if max_truck < no_truck:
				max_truck = no_truck
		if len(truck_route) > 0:
			day_schedule.append(truck_route.copy())
		if len(day_schedule) > 0:
			details.append(day_schedule.copy())
		truck_info['no_truck_used'] = max_truck
		return (truck_info, details)

def main():
	factory = Factory('test_1.txt', seed=200)
	t1 = factory.truckScheduleInit()
	print(t1)
	cost = factory.truckCost(t1)
	print(cost)
	info, detail = factory.sumTruckSchedule(t1)
	print(info)
	for index, day in enumerate(detail):
		print(f"day {index + 1}")
		for truck in day:
			print(truck)

if __name__ == "__main__":
	main()