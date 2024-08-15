#!/usr/bin/env python

import argparse
import itertools as it
from matplotlib import pyplot as plt
import numpy as np
from ortools.sat.python import cp_model
import pathlib as paths
from PIL import Image, ImageOps, ImageGrab
import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
rng = np.random.default_rng(seed=2024)

N_STA = int(1e4)    # how many starting places to try
N_NEI = 16          # how many neighbouring pixels to sample
D_NEI = 4           # how far to look for neighbours
TOO_CLOSE = 3       # same-colour threshold
PAL_WID = 32        # palette pixel width per colour

def main():
    ap = argparse.ArgumentParser(description="Extract and solve LinkedIn Queens puzzles")
    gp = ap.add_mutually_exclusive_group(required=True)
    gp.add_argument(
        '--directory', '-d', type=paths.Path,
        help="Look through a directory and process all PNG images"
    )
    gp.add_argument(
        '--from-clipboard', '-c', action='store_true',
        help="Attempt to process the image in the clipboard"
    )
    ap.add_argument(
        '--sample_size', '-s', type=int, default=10,
        help="Maximum number of images to process when using the --directory option"
    )
    ap.add_argument(
        '--save-file', '-f', type=paths.Path,
        help="Save image file when using the --directory option"
    )

    args = ap.parse_args()

    if args.directory:
        processFilesIn(args.directory, max_sample=args.sample_size, save_as=args.save_file)
    elif args.from_clipboard:
        solveFromClipboard()
    else:
        print("Nothing to do")
                       


def solveBoard(board):
    """Return the column where the queen should be for each row"""

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

    # only allow the queen to be placed if the colour matches the board
    for r in range(n):
        model.add_allowed_assignments(
            [queens[r], q_cols[r]],
            [[c,board[r][c]] for c in range(n)]
        )
        
    solver = cp_model.CpSolver()
    status_code = solver.solve(model)
    if status_code in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return [solver.value(q) for q in queens]
    else:
        return []



def learnPalette(p, n_starts=N_STA, n_neighbours=N_NEI, dist_neighbours=D_NEI,
                 same_thresh=TOO_CLOSE):
    """Use the pixels in an image to work out the 'real' colours in a puzzle """
    keep = set()
    for _ in tqdm.tqdm(range(n_starts),leave=False):
        homeidx = tuple([rng.integers(x) for x in p.shape[:2]])
        homecol = p[homeidx]
        if homecol.sum() > 750 or homecol.sum() < 100: # ignore black and white
            continue
        good = True
        for _ in range(n_neighbours):
            newidx = np.array(homeidx) + np.array(rng.integers(dist_neighbours,size=2))
            try:
                newcol = p[tuple(newidx.tolist())]
            except IndexError: # looking outside the grid?
                good=False
                break
            if not np.array_equal(homecol,newcol):
                good=False
                break
        if good:
            keep.add(tuple(homecol.tolist()))
    # mark as a duplicate any colour which is close enough to another colour
    dupes = set(
        [b for a, b in it.combinations(keep,2)
         if np.max(np.abs(np.array(a)-np.array(b))) < same_thresh]
    )
    cols = keep - dupes
    print(f"Found {len(keep)}; Kept = {len(cols)}")
    return cols


def drawPuzzleAndPalette(p,cols,title='',save_as=None):
    """Show the puzzle and the extracted palette"""
    pal = np.zeros(shape=(PAL_WID,len(cols)*PAL_WID,3),dtype=np.int8)
    for j, (r,g,b) in enumerate(sorted(cols)):
        pal[:,j*PAL_WID:(j+1)*PAL_WID] = np.array([r,g,b])
    fig,axs = plt.subplots(2,1,height_ratios=(len(cols),1),figsize=(2.4,3))
    plt.rcParams.update({'font.size': 8})
    ax = axs[0]
    ax.imshow(Image.fromarray(p))
    ax.set_axis_off()
    ax.set_title(title)
    ax = axs[1]
    ax.set_axis_off()
    ax.set_title(f"Palette, size {len(cols)}")
    ax.imshow(Image.fromarray(pal,mode='RGB'))
    fig.tight_layout()
    fig.savefig(save_as, dpi=600)


def makeConstraintInstance(p,cols):
    """Generate a row,column parameter array to describe a given puzzle """
    # grab the central pixel of each cell
    n = len(cols)
    w = p.shape[0] / n
    centres = np.array([ [p[int(w/2+w*r)][int(w/2+w*c)]  for c in range(n)] for r in range(n)])
    # fix colours to nearest known
    for r,c in it.product(range(n),repeat=2):
        centres[r,c] = sorted(cols, key=lambda rgb:np.sum(np.abs(centres[r,c] - rgb)))[0]
    centre_cols = [x.tolist() for x in np.unique(np.array(centres).reshape(-1,3),axis=0)]
    assert len(centre_cols)==n, "Mismatch between colours in centres vs expected palette"
    board = [ [centre_cols.index(c.tolist()) for c in r] for r in centres]
    print(np.array(board))
    return board


def getEprimeParam(board):
    n = len(board)
    print(f"letting n = {n}\nletting board={board}")
    

def processFilesIn(img_dir, max_sample=-1, save_as=None):
    puzzles = []
    for f_img in list(img_dir.glob("*png")):
        with Image.open(f_img) as im:
            small = ImageOps.contain(im, (128,128)).convert('RGBA')
            pixels = np.array(small)
            puzzles.append(pixels[:,:,:3])
    print(f"Found {len(puzzles)} puzzles.")

    _size = max_sample if max_sample > 0 else len(puzzles)
    puzzle_idxs = np.random.choice(range(len(puzzles)),replace=False,size=_size)
    for i, p in enumerate([puzzles[i] for i in puzzle_idxs]):
        print(f"=== Doing puzzle #{i:2d} === ")
        cols = learnPalette(p)
        if save_as:
            fname = save_as.parent / f"{save_as.stem}-{i}{save_as.suffix}"
            drawPuzzleAndPalette(p, cols, title=f"Board {i:>2d}", save_as=fname)
        params = makeConstraintInstance(p, cols)
        q_pos = solveBoard(params)
        print(f"Solution:\n {np.array(q_pos)}")


def solveFromClipboard():
    img = ImageGrab.grabclipboard()
    print(f"In clipboard, we found:\n{img}")
    puz = np.array(ImageOps.contain(img, (128,128)).convert('RGBA'))[:,:,:3]
    cols = learnPalette(puz)
    params = makeConstraintInstance(puz, cols)
    q_pos = solveBoard(params)
    print(f"Solution:\n {np.array(q_pos)}")
    



    
if __name__=='__main__':
    main()
