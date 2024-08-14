import itertools as it
from ortools.sat.python import cp_model


board = [
    [3, 3, 3, 3, 3, 3, 3, 3],
    [3, 7, 7, 7, 7, 7, 7, 3],
    [0, 2, 7, 7, 7, 7, 5, 4],
    [0, 2, 2, 7, 7, 5, 5, 4],
    [0, 0, 0, 7, 7, 5, 4, 4],
    [1, 1, 7, 7, 7, 7, 4, 6],
    [1, 7, 7, 7, 7, 7, 7, 6],
    [1, 1, 1, 1, 6, 6, 6, 6]
]

n = len(board)
_space = lambda dims: it.product(range(n), repeat=dims)

model  = cp_model.CpModel()

# where is the queen in each row?
queens = [model.new_int_var(0,n-1,f'q_{row:02d}') for row in range(n)]
model.add_all_different(queens)

# queens in adjacent rows must be at least 2 apart
for i in range(len(queens)-1):
    q1,q2 = queens[i:i+2]
    _v = model.new_int_var(0,n, f'diff_{q1}_{q2}')
    model.add_abs_equality(_v, q1-q2)
    model.add_linear_constraint(_v, 2, n)

# what colour is the queen occupying in each row?
q_cols = [model.new_int_var(0,n-1,f'c_{row:02d}') for row in range(n)]
model.add_all_different(q_cols)

for r in range(n):
    model.add_allowed_assignments(
        [queens[r], q_cols[r]],
        [[c,board[r][c]] for c in range(n)]
    )

    
solver = cp_model.CpSolver()
status_code = solver.solve(model)
status_name = solver.status_name()

print(f"Result: {status_name}")
if status_code in (cp_model.FEASIBLE, cp_model.OPTIMAL):
    print('Board:')
    print('\n'.join(map(str,board)))
    print('Solution (queens):')
    print([solver.value(q) for q in queens])
    print('Solution (colours):')
    print([solver.value(v) for v in q_cols])
