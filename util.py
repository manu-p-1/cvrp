"""
util.py

Contains classes and function that act as utilities for the main cvrp problem in cvrp.py
"""
import math
from typing import Union


class Building:

    def __init__(self, node: Union[str, int], x: float, y: float, quant: int):
        """
        Creates a new House object with the parameters
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
        A getter function to return the number of packages to pickup from the building
        :return: The number of packages to pickup from this building
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
        return round(math.sqrt(((b1.x - b2.x) ** 2) + ((b1.y - b2.y) ** 2)))

    def __str__(self):
        return f"Node: {self.node}\nx: {self.x}\ny: {self.y}\nquant: {self.quant}"

    def __repr__(self):
        return f"util.House(node: {self.node}, x: {self.x}, y: {self.y}, quant: {self.quant})"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._node == other.node
        return False


class Vehicle:
    def __init__(self, capacity: int):
        """
        Creates a new Vehicle object - all vehicles in this problem should be homogeneous
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


def populate_from_file(filename: str):
    class Parser:
        def __init__(self):
            self.depot = None
            self.buildings = []

    p = Parser()
    with open(filename, "r") as f:
        for idx, line in enumerate(f):
            ident, x, y, quant = line.split()
            h = Building(int(ident), float(x), float(y), int(quant))

            if idx == 0:
                p.depot = h
            else:
                p.buildings.append(h)
    return p
