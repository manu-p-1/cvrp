"""
util.py

Contains classes and function that act as utilities for the main cvrp problem in cvrp.py
"""
import math


class Building:

    def __init__(self, ident: int, x: int, y: int, num_packages: int):
        """
        Creates a new House object with the parameters
        :param ident: The identity of this house - a unique number
        :param x: The x-coordinate this house sits on
        :param y: The y-coordinate this house sits on
        :param num_packages: The number of packages this house has
        """
        self._x = x
        self._y = y
        self._num_packages = num_packages
        self._ident = ident

    @property
    def x(self) -> int:
        """
        A getter function to return the x-coordinate of this house
        :return: The x-coordinate of this house
        """
        return self._x

    @x.setter
    def x(self, x: int) -> None:
        """
        A setter function to set the x-coordinate for this house
        :param x: Sets the x-coordinate for this house
        :return: None
        """
        self._x = x

    @property
    def y(self) -> int:
        """
        A getter function to return the y-coordinate of this house
        :return: The y-coordinate of this house
        """
        return self._y

    @y.setter
    def y(self, y: int) -> None:
        """
        A setter function to set the y-coordinate for this house
        :param y: Sets the y-coordinate for this house
        :return: None
        """
        self._y = y

    @property
    def num_packages(self) -> int:
        """
        A getter function to return the num_packages of the house
        :return: The num_packages of this house
        """
        return self._num_packages

    @num_packages.setter
    def num_packages(self, capacity: int) -> None:
        """
        A setter function to set the num_packages of the house
        :param capacity: The num_packages to set for the house
        :return: None
        """
        self._num_packages = capacity

    @property
    def ident(self) -> int:
        """
        A getter function to return the identity of the house
        :return: The identity of this house
        """
        return self._ident

    @ident.setter
    def ident(self, ident: int) -> None:
        """
        A setter function to set the identity of the house
        :param ident: The identity to set for the house
        :return: None
        """
        self._ident = ident

    @staticmethod
    def distance(house1: 'Building', house2: 'Building'):
        return math.sqrt((house1.x - house2.x) ** 2 + (house1.y - house2.y) ** 2)

    def __str__(self):
        return f"Identity: {self.ident}\nx: {self.x}\ny: {self.y}\nnum_packages: {self.num_packages}"

    def __repr__(self):
        return f"util.House(ident: {self.ident}, x: {self.x}, y: {self.y}, num_packages: {self.num_packages})"


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
        A getter function to return the num_packages of this vehicle
        :return: The num_packages of this vehicle
        """
        return self._capacity

    @capacity.setter
    def capacity(self, capacity: int) -> None:
        """
        A setter function to set the num_packages for this vehicle
        :param capacity: Sets the num_packages for this vehicle
        :return: None
        """
        self._capacity = capacity


def populate_from_file(filename: str):

    ll = []
    with open(filename, "r") as f:
        for line in f:
            spt = [int(x) for x in line.split()]
            h = Building(spt[0], spt[1], spt[2], spt[3])
            ll.append(h)
    return ll
