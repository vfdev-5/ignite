from ignite.base import EventsDriven, Serializable
from ignite.engine.events import EventEnum


def test_load_state_dict():

    s = Serializable()
    s.load_state_dict({})


class ABCEvents(EventEnum):
    A_EVENT = "A_event"
    B_EVENT = "B_event"
    C_EVENT = "C_event"


def test_events_driven_basics():

    e = EventsDriven()
    assert len(e._allowed_events) == 0

    e.register_events("a", "b", "c", *ABCEvents)

    times_said_hello = [0]

    @e.on("a")
    def say_hello():
        times_said_hello[0] += 1

    e.fire_event("a")
    e.fire_event("a")
    assert times_said_hello[0] == 2

    times_handled_b_event = [0]

    def on_b_event():
        times_handled_b_event[0] += 1

    e.add_event_handler(ABCEvents.B_EVENT(every=2), on_b_event)

    e.fire_event(ABCEvents.B_EVENT)
    e.fire_event(ABCEvents.B_EVENT)
    e.fire_event(ABCEvents.B_EVENT)
    e.fire_event(ABCEvents.B_EVENT)
    e.fire_event(ABCEvents.A_EVENT)
    e.fire_event(ABCEvents.A_EVENT)
    e.fire_event(ABCEvents.C_EVENT)
    assert times_handled_b_event[0] == 2
    assert e._allowed_events_counts[ABCEvents.A_EVENT] == 2
    assert e._allowed_events_counts[ABCEvents.B_EVENT] == 4
    assert e._allowed_events_counts[ABCEvents.C_EVENT] == 1


def test_tiny_engine():
    class State:
        def __init__(self):
            self.a = 0
            self.b = 0
            self.c = 0
            self.d = 0

    class EventsDrivenState:
        def __init__(self, state: State, evd: EventsDriven):
            self._evd = evd

        @property
        def a(self):
            return self._evd._allowed_events_counts[ABCEvents.A_EVENT]

        @property
        def b(self):
            return self._evd._allowed_events_counts[ABCEvents.B_EVENT]

        @property
        def c(self):
            return self._evd._allowed_events_counts[ABCEvents.C_EVENT]

        @a.setter
        def a(self, value):
            self._evd._allowed_events_counts[ABCEvents.A_EVENT] = value

        @b.setter
        def b(self, value):
            self._evd._allowed_events_counts[ABCEvents.B_EVENT] = value

        @c.setter
        def c(self, value):
            self._evd._allowed_events_counts[ABCEvents.C_EVENT] = value

    class TinyEngine(EventsDriven):
        def __init__(self):
            super(TinyEngine, self).__init__()
            self.register_events(*ABCEvents)
            self.state = State(self)

        def _check(self):
            assert self.state.a == self._allowed_events_counts[ABCEvents.A_EVENT]
            assert self.state.b == self._allowed_events_counts[ABCEvents.B_EVENT]
            assert self.state.c == self._allowed_events_counts[ABCEvents.C_EVENT]

        def run(self, n, k, reset=True):
            if reset:
                self._reset_allowed_events_counts()
            self.fire_event(ABCEvents.A_EVENT)
            while self.state.b < n:
                self.fire_event(ABCEvents.B_EVENT)
                j = self.state.c % k
                while j < k:
                    j += 1
                    self.fire_event(ABCEvents.C_EVENT)
                    self._check()

    e = TinyEngine()
    e.run(10, 20)
    assert e.state.a == 1
    assert e.state.b == 10
    assert e.state.c == 20 * 10

    e.state.a = 0
    e.state.b = 5
    e.state.c = 0
    e.run(10, 20, reset=False)
    assert e.state.a == 1
    assert e.state.b == 10
    assert e.state.c == 20 * 5
