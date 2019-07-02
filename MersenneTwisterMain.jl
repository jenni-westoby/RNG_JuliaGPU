using CUDAdrv, CUDAnative, CuArrays
using RandomNumbers.MersenneTwisters

#Small function required by twister
mt_magic(y) = ((y % Int32) << 31 >> 31) & 0x9908b0df

#Function that runs in parallel
function kernel(backing_array, seedArr, outputArray, N_rand_to_make, N)

    #set some constants
    M = 397
    UPPER_MASK = 0x80000000
    LOWER_MASK = 0x7fffffff

    #Set tid and totalThreads
    tid = ((threadIdx().x) + (blockIdx().x - 1) * blockDim().x)
    totalThreads = blockDim().x * gridDim().x

    #Set MT19937 array (mt) and integer (mti) from seedArr[tid]
    backing_array[threadIdx().x, blockIdx().x, 1] = seedArr[tid]
    for i in 2:N
        backing_array[threadIdx().x, blockIdx().x, i] = 0x6c078965 * (backing_array[threadIdx().x, blockIdx().x, i-1] ⊻ (backing_array[threadIdx().x, blockIdx().x, i-1] >> 30)) + (i - 1) % UInt32
    end
    mti = N + 1

    #This while loop allows more random numbers to be generated than there are threads initialised on the GPU
    while tid <= N_rand_to_make
        ############## I do this inline so I can mutate MT19937 directly
        if mti > N
            for i in 1:N-M
                y = (backing_array[threadIdx().x, blockIdx().x, i] & UPPER_MASK) | (backing_array[threadIdx().x, blockIdx().x, i+1] & LOWER_MASK)
                backing_array[threadIdx().x, blockIdx().x, i] = backing_array[threadIdx().x, blockIdx().x, i + M] ⊻ (y >> 1) ⊻ mt_magic(y)
            end
            for i in N-M+1:N-1
                y = (backing_array[threadIdx().x, blockIdx().x, i] & UPPER_MASK) | (backing_array[threadIdx().x, blockIdx().x, i+1] & LOWER_MASK)
                backing_array[threadIdx().x, blockIdx().x, i] = backing_array[threadIdx().x, blockIdx().x, i + M - N] ⊻ (y >> 1) ⊻ mt_magic(y)
            end
            begin
                y = (backing_array[threadIdx().x, blockIdx().x, N] & UPPER_MASK) | (backing_array[threadIdx().x, blockIdx().x, 1] & LOWER_MASK)
                backing_array[threadIdx().x, blockIdx().x, N] = backing_array[threadIdx().x, blockIdx().x, M] ⊻ (y >> 1) ⊻ mt_magic(y)
            end
            mti = 1
        end
        k = backing_array[threadIdx().x, blockIdx().x, mti]
        k ⊻= (k >> 11)
        k ⊻= (k << 7) & 0x9d2c5680
        k ⊻= (k << 15) & 0xefc60000
        k ⊻= (k >> 18)

        mti += 1
        ############
        outputArray[tid] = k
        tid += totalThreads
    end

    return nothing
end

function gpuRand(threadNum, blockNum, randNum)

    #Make an array of random states to seed with
    N = 624
    seed_arr = []
    for i in 1:((randNum ÷ N) + 1)
        append!(seed_arr, MT19937().mt)

    end
    seed_array = CuArrays.CuArray(UInt32.(seed_arr))

    #Make other args
    outputArray = CuArrays.CuArray(fill(0, randNum))
    backing_array = CuArrays.CuArray(fill(UInt32(0), threadNum, blockNum, N))

    #Make random numbers on GPU
    @cuda threads = threadNum blocks = blockNum kernel(backing_array, seed_array, outputArray, randNum, N)

    return Array(outputArray)
end
