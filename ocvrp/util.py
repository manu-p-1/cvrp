"""
https://github.com/manu-p-1/cvrp
util.py

This module contains utility functions and classes for the CVRP problem optimization
"""
import collections
import math
from json import JSONEncoder
from typing import Tuple, List, Union


class Building:

    def __init__(self, node: int, x: float, y: float, quant: int):
        """
        Creates a new Building instance with the parameters

        :param node: The node number of this house - a unique number
        :param x: The x-coordinate this house sits on
        :param y: The y-coordinate this house sits on
        :param quant: The number of packages this house has
        """
        self._x = x
        self._y = y
        self._quant = quant
        self._node = node

    @property
    def x(self) -> float:
        """
        A getter function to return the x-coordinate of this house
        :return: The x-coordinate of this house
        """
        return self._x

    @x.setter
    def x(self, x: float) -> None:
        """
        A setter function to set the x-coordinate for this house
        :param x: Sets the x-coordinate for this house
        :return: None
        """
        self._x = x

    @property
    def y(self) -> float:
        """
        A getter function to return the y-coordinate of this house
        :return: The y-coordinate of this house
        """
        return self._y

    @y.setter
    def y(self, y: float) -> None:
        """
        A setter function to set the y-coordinate for this house
        :param y: Sets the y-coordinate for this house
        :return: None
        """
        self._y = y

    @property
    def quant(self) -> int:
        """
        A getter function to return the service demand for this building
        :return: The service demand for this building
        """
        return self._quant

    @quant.setter
    def quant(self, capacity: int) -> None:
        """
        A setter function to set the number of packages to pickup from the building
        :param capacity: The number of packages to pickup from this building
        :return: None
        """
        self._quant = capacity

    @property
    def node(self) -> int:
        """
        A getter function to return the node number of the house
        :return: The node number of this house
        """
        return self._node

    @node.setter
    def node(self, ident: int) -> None:
        """
        A setter function to set the node number of the house

        :param ident: The node number to set for the house
        :return: None
        """
        self._node = ident

    @staticmethod
    def distance(b1: 'Building', b2: 'Building'):
        """
        Calculates the euclidean distance between two Building instances

        :param b1: The first Building
        :param b2: The second Building
        :return: The euclidean distance between the Buildings, rounded to the nearest integer
        """
        return round(math.sqrt(((b1.x - b2.x) ** 2) + ((b1.y - b2.y) ** 2)))

    def __str__(self):
        return f"Node: {self.node}, x: {self.x}, y: {self.y}, quant: {self.quant}"

    def __repr__(self):
        return f"util.Building<node: {self.node}, x: {self.x}, y: {self.y}, quant: {self.quant}>"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._node == other.node
        return False

    def __key(self) -> Tuple:
        return self._node, self._x, self._y, self._quant

    def __hash__(self):
        return hash(self.__key())


class CVRPEncoder(JSONEncoder):
    """
    Creates a JSONEncoder child class which allows a CVRP instance to be converted to JSON
    """

    def default(self, o):
        if isinstance(o, Building):
            return o.__dict__


class Individual(collections.abc.Sequence):

    def __init__(self, genes, fitness: Union[None, float]):
        """
        Creates a new Individual instance that associates a set of genes with a fitness value
        :param genes: The genes (A list of Building objects) representing the genetic fragment
        :param fitness: The fitness value of the genes. Fitness may be empty at various stages of the algorithm
        """
        self._genes = genes
        self._fitness = fitness

    @property
    def genes(self) -> List[Building]:
        """
        A getter function to return the capacity of this vehicle
        :return: The genes for this Individual
        """
        return self._genes

    @genes.setter
    def genes(self, genes: List[Building]) -> None:
        """
        A setter function to set the genes for this Individual
        :param genes: Sets the genes for this Individual
        :return: None
        """
        self._genes = genes

    @property
    def fitness(self) -> Union[float, None]:
        """
        A getter function to return the fitness of this Individual
        :return: The the fitness of this Individual
        """
        return self._fitness

    @fitness.setter
    def fitness(self, fitness: float) -> None:
        """
        A setter function to set the fitness for this Individual
        :param fitness: Sets the fitness for this Individual
        :return: None
        """
        self._fitness = fitness

    def __key(self) -> Tuple:
        return (self._fitness,)

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        return f"Genes: {self._genes},\nfitness: {self._fitness}"

    def __repr__(self):
        return f"util.Individual<genes: {self._genes}, fitness: {self._fitness}>"

    def __iter__(self):
        for g in self._genes:
            yield g

    def __contains__(self, item):
        return item in self._genes

    def __getitem__(self, item):
        return self._genes[item]

    def __setitem__(self, key, value):
        self._genes[key] = value

    def __delitem__(self, key):
        self.__delattr__(key)

    def __len__(self):
        return len(self._genes)

    def __eq__(self, other):
        return self._fitness == other.fitness

    def __ne__(self, other):
        return self._fitness != other.fitness

    def __lt__(self, other):
        return self._fitness < other.fitness

    def __le__(self, other):
        return self._fitness <= other.fitness

    def __ge__(self, other):
        return self._fitness >= other.fitness

    def __gt__(self, other):
        return self._fitness > other.fitness

    def __radd__(self, other):
        return other + self._fitness


class Vehicle:
    def __init__(self, capacity: int):
        """
        Creates a new Vehicle object - all vehicles in this problem should be of homogeneous capacity
        :param capacity: The capacity this vehicle can handle
        """
        self._capacity = capacity

    @property
    def capacity(self) -> int:
        """
        A getter function to return the capacity of this vehicle
        :return: The the capacity of this vehicle
        """
        return self._capacity

    @capacity.setter
    def capacity(self, capacity: int) -> None:
        """
        A setter function to set the capacity for this vehicle
        :param capacity: Sets the capacity for this vehicle
        :return: None
        """
        self._capacity = capacity


class OCVRPParser:
    """
    The parser for the .ocvrp type. The file is made of header values in the format:
    HEADER:VALUE
    
    There are 6 known header values: NAME, COMMENTS, DIM, CAPACITY, OPTIMAL, and NODES
    
    Spacing in between header values is irrelevant including the order in which the values are written.
    The tabular numeric data for all the nodes must be written under the NODES header. More information
    on the format is available in the README.md

    The primary motivation behind creating our own data format and parser was the unintuitive format of
    most VRP data sets. They are difficult to parse and nodes and service demands are complicated to discern.
    The .ocvrp format is straightforward, easy to read, and flexible. In theory, the .ocvrp format could be
    extensible for time windowed applications too.
    """

    def __init__(self, filename):
        """
        Creates a new OCVRPParser with the filename
        :param filename: The filename containing the .ocvrp file and dataset
        """
        if not filename.endswith(".ocvrp"):
            raise SyntaxError("File is not of .ocvrp type")

        self.f = open(filename, "r")
        self._values = {}
        self._headers = ("NAME", "COMMENTS", "DIM", "CAPACITY", "OPTIMAL")
        self._num_headers = ("DIM", "CAPACITY", "OPTIMAL")

    class __OCVRPParserStrategy:

        def __init__(self, parser):
            self.__parser = parser

        def get_ps_depot(self):
            """
            Returns the depot location of the problem set
            :return: The depot location of the problem set as a Building object
            """
            return self.__parser._values["DEPOT"]

        def get_ps_buildings(self):
            """
            Returns the population for this instance
            :return: The entire population for this instance as a list of Individual instances
            """
            return self.__parser._values["BUILDINGS"]

        def get_ps_name(self):
            """
            Returns the name of the problem set
            :return: The name of the problem set for this instance
            """
            return self.__parser._values["NAME"]

        def get_ps_comments(self):
            """
            Returns the comments of the problem set
            :return: The comments of the problem set for this instance
            """
            return self.__parser._values["COMMENTS"] if "COMMENTS" in self.__parser._values else None

        def get_ps_dim(self):
            """
            Returns the working dimension of the problem set (not including the depot)
            :return: The working dimension of the problem set for this instance
            """
            return self.__parser._values["DIM"] - 1

        def get_ps_capacity(self):
            """
            Returns the vehicle capacity of the problem set
            :return: The vehicle capacity of the problem set for this instance
            """
            return self.__parser._values["CAPACITY"]

        def get_ps_optimal(self):
            """
            Returns the optimal fitness of the problem set
            :return: The optimal fitness of the problem set for this instance
            """
            return self.__parser._values["OPTIMAL"]

    def parse(self):
        """
        Parses the dataset with the dataset loaded on this instance
        :return: An OCVRPStrategy object containing getters for problem set headers
        """
        lines = self.f.readlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if line not in ('\n', '\r\n'):
                try:
                    ln = line.split(":")
                    ln0 = ln[0].upper()
                    ln1 = ln[1]

                    # If the header is the node, we need to load all the nodes underneath the header
                    if ln0 == 'NODES':
                        idx = self._grab_buildings(idx + 1, lines)
                    else:
                        if self._is_number(ln1) and ln0 in self._num_headers:
                            # Load the header as an integer if it's numeric
                            self._values[ln0] = int(ln1)
                        else:
                            self._values[ln0] = ln1.replace("\n", "").strip()
                except Exception as e:
                    raise SyntaxError("File is not formatted properly", e)
            idx += 1

        self.f.close()
        return self.__OCVRPParserStrategy(self)

    def _grab_buildings(self, curr, lines):
        """
        Creates Building objects from all nodes underneath the NODES header.
        :param curr: The current index of the first node
        :param lines: The entire .ocvrp file as a readlines() list
        :return: The current index in the file after loading all Nodes
        """
        ll = len(lines)
        buildings = []
        ctr = 0

        while curr < ll:
            line = lines[curr]
            ls = line.split()

            ident, x, y, quant = ls
            h = Building(int(ident), float(x), float(y), int(quant))

            # First node is always DEPOT
            if ctr == 0:
                self._values["DEPOT"] = h
            else:
                buildings.append(h)

            # Check if EOF or if the next line is numeric (if not, indicates that node parsing is done)
            if curr < ll - 1:
                next_num = lines[curr + 1].split()
                if len(next_num) == 0 or not self._is_number(next_num[0]):
                    break
            else:
                break

            ctr += 1
            curr += 1

        self._values["BUILDINGS"] = buildings
        return curr

    @staticmethod
    def _is_number(num: str):
        """
        Given a string, returns whether it is a number with unicode sensitivity
        :param num: The number as a string
        :return: If num is a numerical value
        """
        try:
            float(num)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(num)
            return True
        except (TypeError, ValueError):
            pass

        return False
