from time import sleep

VALUES = []
def vals():
    for i in range(4):
        yield(i)
        sleep(i)

print(f"VALUES: {VALUES}")
gens = vals()
for v in gens:
    print(v)
    VALUES.append(v)
print(f"AFTER: {VALUES}")
