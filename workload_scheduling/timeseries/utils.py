from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import threading
import heapq
import time
from pydantic import BaseModel

@dataclass(order=True)
class PrioritizedQuery:
    priority: float
    query: 'Query' = field(compare=False)
    enqueue_time: float = field(compare=False, default_factory=time.time)
    start_time: Optional[float] = field(compare=False, default=None)
    retry_count: int = field(compare=False, default=0)
    next_available_time: float = field(compare=False, default_factory=lambda: time.time())

@dataclass
class Query:
    id: int
    sql: str
    explain_json_plan: Dict[str, Any]
    submit_time: float = field(default_factory=time.time)
    end_time: float = field(default=None)

# ----------------------------
# Priority Queue Implementation
# ----------------------------
class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.lock = threading.Lock()
    
    def push(self, prioritized_query: PrioritizedQuery):
        with self.lock:
            heapq.heappush(self.heap, prioritized_query)
    

    def pop_ready_queries(self, current_time: float) -> List[PrioritizedQuery]:
        ready = []
        with self.lock:
            while self.heap and self.heap[0].next_available_time <= current_time:
                ready.append(heapq.heappop(self.heap))
        return ready    

    def peek_next_available_time(self) -> Optional[float]:
        with self.lock:
            if self.heap:
                return self.heap[0].next_available_time
            return None
    
    def is_empty(self) -> bool:
        with self.lock:
            return len(self.heap) == 0

# Define the request model
class QueryRequest(BaseModel):
    id: int
    sql: str
    explain_json_plan: Dict[str, Any]