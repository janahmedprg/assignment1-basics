class BytePairMaxHeap:

    def __init__(self, byte_pairs_freq: dict[tuple[bytes, bytes], int]):
        self.bp_heap = list(byte_pairs_freq.items())
        
        i = (len(self.bp_heap)-1)//2

        while i >= 0:
            _heapify(self.bp_heap, i)
            i -= 1
    
    def push(self, node):
        self.bp_heap.append(node)
        _siftup(self.bp_heap, len(self.bp_heap) - 1)
    
    def pop(self):
        _swap(self.bp_heap, len(self.bp_heap) - 1, 0)
        ret = self.bp_heap.pop()
        _heapify(self.bp_heap, 0)
        return ret
    
def _swap(bp_heap, i, j):
    bp_heap[i], bp_heap[j] = bp_heap[j], bp_heap[i]

def _heapify(bp_heap, parent):
    child = 2 * parent + 1
    if child >= len(bp_heap):
        return

    if (
        child + 1 < len(bp_heap) and
        (
            bp_heap[child + 1][1] > bp_heap[child][1] or
            (
                bp_heap[child + 1][1] == bp_heap[child][1] and
                bp_heap[child + 1][0] > bp_heap[child][0]
            )
        )
    ):
        child += 1
    
    if (
        bp_heap[child][1] > bp_heap[parent][1] or
        (
            bp_heap[child][1] == bp_heap[parent][1] and
            bp_heap[child][0] > bp_heap[parent][0]
        )
    ):
        _swap(bp_heap, child, parent)
        _heapify(bp_heap, child)

def _siftup(bp_heap, node):
    parent = (node - 1)//2

    if parent < 0:
        return

    if (
        bp_heap[node][1] > bp_heap[parent][1] or
        (
            bp_heap[node][1] == bp_heap[parent][1] and
            bp_heap[node][0] > bp_heap[parent][0]
        )
    ):
        _swap(bp_heap, node, parent)
        _siftup(bp_heap, parent)