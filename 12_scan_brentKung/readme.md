## Scan brent-kung method
- to address low work efficiency of koggleStone scan

- kogge-stone
a. Log(N) steps b. O(N*logN) operations

- brent-kung
a. reduction step
- LogN steps
- N-1 operations

b. post reduction step
- logN - 1 steps
- N-2 - (LogN - 1)

c. Total
- 2*log(N) - 1steps
- O(n) operations

- takes more steps but is more work efficient


## optmizations similar to kogge-stone
1. shared memory: data reuse, memory coalescings
2. doubleBuffering : is not needed

problem:
- memory was coalesced in kogge-stone, therefore using shared memory is even more effective
- doubleBUffering is not needed as dont write and read the same element in the same iterations therefore single iteration is okay
- control divergence is a problem in brentKung unlike koggeStone

## SOlution to minimize control divergence
- do not assign threads to specific data elements
- re-index threads on every tieration to differnet elements(general approach to reduce control divergence)
-
