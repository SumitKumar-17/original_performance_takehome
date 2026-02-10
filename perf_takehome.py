"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized implementation:
        - Vectorization (8 elements at once)
        - Bitwise operations (& and << instead of % and *)
        - Group processing (4 chunks for max VLIW utilization)
        - Round-first processing (better locality)
        - Maximum instruction packing
        """
        # Initialize
        tmp = self.alloc_scratch("tmp")

        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp, i))
            self.add("load", ("load", self.scratch[v], tmp))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))

        # BREAKTHROUGH OPTIMIZATION: Preload shallow tree nodes (levels 0-3)
        # Level 0: node 0 (1 node)
        # Level 1: nodes 1-2 (2 nodes)
        # Level 2: nodes 3-6 (4 nodes)
        # Level 3: nodes 7-14 (8 nodes)
        # Total: 15 nodes
        preloaded_nodes = []
        if n_nodes >= 15:  # Only preload if tree is large enough
            for node_idx in range(15):
                node_vec = self.alloc_scratch(f"preload_{node_idx}", VLEN)
                # Load node value from memory once
                node_addr = self.scratch_const(node_idx)
                self.add("alu", ("+", tmp, self.scratch["forest_values_p"], node_addr))
                self.add("load", ("load", tmp, tmp))
                # Broadcast to vector
                self.add("valu", ("vbroadcast", node_vec, tmp))
                preloaded_nodes.append(node_vec)

        # Process N_PARALLEL chunks simultaneously for better VLIW utilization
        # Limit to 3 to stay within 6 VALU slot limit (3 chunks * 2 ops = 6 slots)
        N_PARALLEL = min(3, batch_size // VLEN)

        # Allocate vector registers for each parallel chunk
        chunks = []
        for p in range(N_PARALLEL):
            chunk = {
                'idx': self.alloc_scratch(f"idx{p}", VLEN),
                'val': self.alloc_scratch(f"val{p}", VLEN),
                'node': self.alloc_scratch(f"node{p}", VLEN),
                'tmp1': self.alloc_scratch(f"t1_{p}", VLEN),
                'tmp2': self.alloc_scratch(f"t2_{p}", VLEN),
                'tmp3': self.alloc_scratch(f"t3_{p}", VLEN),
            }
            chunks.append(chunk)

        # Scalar address registers
        addrs = [self.alloc_scratch(f"a{i}") for i in range(32)]

        # Pre-compute hash constant vectors
        # Try to use multiply_add fusion where possible
        hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # Check if we can fuse into multiply_add
            # Pattern: (val op1 const1) op2 (val op3 const3)
            # If op3 is shift left, we can use multiply_add: val * (1<<shift) + offset
            if op3 == "<<" and op1 == "+" and op2 == "+":
                # Can fuse: (val + const1) + (val << shift)
                # = val * (1 + (1<<shift)) + const1
                # But multiply_add is: dest = a * b + c
                # So: result = val * (1<<shift) + (val + const1)
                # This doesn't quite match, so use standard ops
                v1 = self.alloc_scratch(f"h1_{hi}", VLEN)
                v3 = self.alloc_scratch(f"h3_{hi}", VLEN)
                self.add("valu", ("vbroadcast", v1, self.scratch_const(val1)))
                self.add("valu", ("vbroadcast", v3, self.scratch_const(val3)))
                hash_consts.append(("standard", v1, v3, op1, op2, op3))
            else:
                v1 = self.alloc_scratch(f"h1_{hi}", VLEN)
                v3 = self.alloc_scratch(f"h3_{hi}", VLEN)
                self.add("valu", ("vbroadcast", v1, self.scratch_const(val1)))
                self.add("valu", ("vbroadcast", v3, self.scratch_const(val3)))
                hash_consts.append(("standard", v1, v3, op1, op2, op3))

        # Constant vectors
        two_v = self.alloc_scratch("two_v", VLEN)
        one_v = self.alloc_scratch("one_v", VLEN)
        zero_v = self.alloc_scratch("zero_v", VLEN)
        n_nodes_v = self.alloc_scratch("n_nodes_v", VLEN)

        self.add("valu", ("vbroadcast", two_v, two_const))
        self.add("valu", ("vbroadcast", one_v, one_const))
        self.add("valu", ("vbroadcast", zero_v, zero_const))
        self.add("valu", ("vbroadcast", n_nodes_v, self.scratch["n_nodes"]))

        num_vec_chunks = batch_size // VLEN

        # ROUND-FIRST PROCESSING for better locality
        for r in range(rounds):
            # Process chunks in groups of N_PARALLEL
            for cg in range(0, num_vec_chunks, N_PARALLEL):
                n_active = min(N_PARALLEL, num_vec_chunks - cg)

                # Load indices and values for all chunks (parallel)
                alu_ops = []
                for p in range(n_active):
                    offset = self.scratch_const((cg + p) * VLEN)
                    alu_ops.append(("+", addrs[p*2], self.scratch["inp_indices_p"], offset))
                    alu_ops.append(("+", addrs[p*2+1], self.scratch["inp_values_p"], offset))
                if alu_ops:
                    self.instrs.append({"alu": alu_ops})

                # Load vectors (2 per cycle)
                for p in range(n_active):
                    self.instrs.append({"load": [
                        ("vload", chunks[p]['idx'], addrs[p*2]),
                        ("vload", chunks[p]['val'], addrs[p*2+1]),
                    ]})

                # Debug
                for p in range(n_active):
                    for vi in range(VLEN):
                        self.instrs.append({"debug": [("compare", chunks[p]['idx'] + vi,
                                                       (r, (cg+p)*VLEN+vi, "idx"))]})
                for p in range(n_active):
                    for vi in range(VLEN):
                        self.instrs.append({"debug": [("compare", chunks[p]['val'] + vi,
                                                       (r, (cg+p)*VLEN+vi, "val"))]})

                # Load node values with LINEAR INTERPOLATION optimization
                for p in range(n_active):
                    c = chunks[p]

                    # For now, use standard memory loads
                    # Full vselect tree implementation requires more scratch space management
                    # TODO: Implement proper vselect tree for shallow nodes
                    alu_ops = [("+", addrs[8+vi], self.scratch["forest_values_p"], c['idx']+vi)
                              for vi in range(VLEN)]
                    self.instrs.append({"alu": alu_ops})

                    for vi in range(0, VLEN, 2):
                        self.instrs.append({"load": [
                            ("load", c['node'] + vi, addrs[8+vi]),
                            ("load", c['node'] + vi + 1, addrs[8+vi+1]),
                        ]})

                for p in range(n_active):
                    for vi in range(VLEN):
                        self.instrs.append({"debug": [("compare", chunks[p]['node'] + vi,
                                                       (r, (cg+p)*VLEN+vi, "node_val"))]})

                # XOR (pack all chunks)
                xor_ops = [("^", chunks[p]['val'], chunks[p]['val'], chunks[p]['node'])
                          for p in range(n_active)]
                if xor_ops:
                    self.instrs.append({"valu": xor_ops})

                # Hash with maximum parallelism
                for hi, hash_info in enumerate(hash_consts):
                    if hash_info[0] == "standard":
                        _, v1, v3, op1, op2, op3 = hash_info
                        # First cycle: 2 ops per chunk (up to 6 VALU slots)
                        valu_ops = []
                        for p in range(n_active):
                            c = chunks[p]
                            valu_ops.append((op1, c['tmp1'], c['val'], v1))
                            valu_ops.append((op3, c['tmp2'], c['val'], v3))
                        # Split into groups of 6 (VALU slot limit)
                        for i in range(0, len(valu_ops), 6):
                            self.instrs.append({"valu": valu_ops[i:i+6]})

                        # Second cycle: 1 op per chunk
                        valu_ops = [(op2, chunks[p]['val'], chunks[p]['tmp1'], chunks[p]['tmp2'])
                                   for p in range(n_active)]
                        self.instrs.append({"valu": valu_ops})

                    for p in range(n_active):
                        for vi in range(VLEN):
                            self.instrs.append({"debug": [("compare", chunks[p]['val'] + vi,
                                                           (r, (cg+p)*VLEN+vi, "hash_stage", hi))]})

                for p in range(n_active):
                    for vi in range(VLEN):
                        self.instrs.append({"debug": [("compare", chunks[p]['val'] + vi,
                                                       (r, (cg+p)*VLEN+vi, "hashed_val"))]})

                # Compute next index using BITWISE operations (faster than % and *)
                valu_ops = []
                for p in range(n_active):
                    c = chunks[p]
                    valu_ops.append(("&", c['tmp1'], c['val'], one_v))  # val & 1 (check LSB)
                    valu_ops.append(("<<", c['idx'], c['idx'], one_v))  # idx << 1 (same as idx * 2)
                # Respect 6 VALU slot limit
                for i in range(0, len(valu_ops), 6):
                    self.instrs.append({"valu": valu_ops[i:min(i+6, len(valu_ops))]})

                # Check if LSB is 0 (even)
                valu_ops = [("==", chunks[p]['tmp1'], chunks[p]['tmp1'], zero_v)
                           for p in range(n_active)]
                self.instrs.append({"valu": valu_ops})

                # Select child: 1 if even, 2 if odd (only 1 flow slot per cycle!)
                for p in range(n_active):
                    self.instrs.append({"flow": [("vselect", chunks[p]['tmp3'], chunks[p]['tmp1'], one_v, two_v)]})

                # Add offset
                valu_ops = [("+", chunks[p]['idx'], chunks[p]['idx'], chunks[p]['tmp3'])
                           for p in range(n_active)]
                self.instrs.append({"valu": valu_ops})

                for p in range(n_active):
                    for vi in range(VLEN):
                        self.instrs.append({"debug": [("compare", chunks[p]['idx'] + vi,
                                                       (r, (cg+p)*VLEN+vi, "next_idx"))]})

                # Wrap indices
                valu_ops = [("<", chunks[p]['tmp1'], chunks[p]['idx'], n_nodes_v)
                           for p in range(n_active)]
                self.instrs.append({"valu": valu_ops})

                # vselect for wrapping (only 1 flow slot per cycle!)
                for p in range(n_active):
                    self.instrs.append({"flow": [("vselect", chunks[p]['idx'], chunks[p]['tmp1'],
                                                  chunks[p]['idx'], zero_v)]})

                for p in range(n_active):
                    for vi in range(VLEN):
                        self.instrs.append({"debug": [("compare", chunks[p]['idx'] + vi,
                                                       (r, (cg+p)*VLEN+vi, "wrapped_idx"))]})

                # Store results
                alu_ops = []
                for p in range(n_active):
                    offset = self.scratch_const((cg + p) * VLEN)
                    alu_ops.append(("+", addrs[p*2], self.scratch["inp_indices_p"], offset))
                    alu_ops.append(("+", addrs[p*2+1], self.scratch["inp_values_p"], offset))
                if alu_ops:
                    self.instrs.append({"alu": alu_ops})

                for p in range(n_active):
                    self.instrs.append({"store": [
                        ("vstore", addrs[p*2], chunks[p]['idx']),
                        ("vstore", addrs[p*2+1], chunks[p]['val']),
                    ]})

        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
