# class LSH:
#     def __init__(self, front, rear):
#         self.front = front
#         self.rear = rear
#
#     def multiply(self, number):
#         front = self.front * number.rear
#         rear = self.rear * number.front
#         result = LSH(front, rear)
#         return result
#
# n1 = LSH(5, 3)
# n2 = LSH(8, 7)
#
# result = n1.multiply(n2)
# print(result.front)
# print(result.rear)

# class Triangle:
#     def __init__(self, a, b, c):
#         self.a = a
#         self.b = b
#         self.c = c
#
#     def perimeter(self):
#         result = self.a + self.b + self.c
#         return result
#
# t1 = Triangle(3, 4, 5)
# print(t1.perimeter())

def test_generator():
    yield 1
    yield 2
    yield 3

gen = test_generator()
for i in test_generator():
    print(i)