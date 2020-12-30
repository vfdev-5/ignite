import pytest

from ignite.base import EventsDriven, EventsDrivenState, Serializable
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


def test_basic_events_driven_with_state():

    class TinyEngine(EventsDriven):

        def __init__(self):
            super(TinyEngine, self).__init__()
            self.register_events(*ABCEvents)
            self._state = EventsDrivenState(
                engine=self, event_to_attr={
                    ABCEvents.A_EVENT: "a",
                    ABCEvents.B_EVENT: "b",
                    ABCEvents.C_EVENT: "c",
                }
            )

        @property
        def state(self):
            return self._state

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
    e.state.b = 3
    e.state.c = 4

    assert e._allowed_events_counts[ABCEvents.A_EVENT] == 0
    assert e._allowed_events_counts[ABCEvents.B_EVENT] == 3
    assert e._allowed_events_counts[ABCEvents.C_EVENT] == 4

    e.run(10, 20, reset=False)
    assert e.state.a == 1
    assert e.state.b == 10
    assert e.state.c == 20 * (10 - 3)

    with pytest.raises(AttributeError, match=r"can't set attribute"):
        e.state = EventsDrivenState()

    e.state.a = 3
    e.state.b = 4
    e.state.c = 5

    assert e._allowed_events_counts[ABCEvents.A_EVENT] == 3
    assert e._allowed_events_counts[ABCEvents.B_EVENT] == 4
    assert e._allowed_events_counts[ABCEvents.C_EVENT] == 5

    e.run(10, 20, reset=False)
    assert e.state.a == 4
    assert e.state.b == 10
    assert e.state.c == 20 * (10 - 4)


def test_events_driven_with_state_mixed_events():

    class BCEvents(EventEnum):
        B_EVENT_STARTED = "B_event_started"
        B_EVENT_COMPLETED = "B_event_completed"
        C_EVENT_STARTED = "C_event_started"
        C_EVENT_COMPLETED = "C_event_completed"

    class AnotherTinyEngine(EventsDriven):

        def __init__(self):
            super(AnotherTinyEngine, self).__init__()
            self.register_events(*BCEvents)
            self._state = EventsDrivenState(
                engine=self, event_to_attr={
                    BCEvents.B_EVENT_STARTED: "b",
                    BCEvents.C_EVENT_STARTED: "c",
                    BCEvents.B_EVENT_COMPLETED: "b",
                    BCEvents.C_EVENT_COMPLETED: "c",
                }
            )

        @property
        def state(self):
            return self._state

        def _check(self):
            assert self.state.b == self._allowed_events_counts[BCEvents.B_EVENT_STARTED]
            assert self.state.c == self._allowed_events_counts[BCEvents.C_EVENT_STARTED]
            assert self.state.b - 1 == self._allowed_events_counts[BCEvents.B_EVENT_COMPLETED]
            assert self.state.c - 1 == self._allowed_events_counts[BCEvents.C_EVENT_COMPLETED]

        def run(self, n, k, reset=True):
            if reset:
                self._reset_allowed_events_counts()
            while self.state.b < n:
                self.fire_event(BCEvents.B_EVENT_STARTED)
                j = self.state.c % k
                while j < k:
                    j += 1
                    self.fire_event(BCEvents.C_EVENT_STARTED)
                    self._check()
                    self.fire_event(BCEvents.C_EVENT_COMPLETED)
                self.fire_event(BCEvents.B_EVENT_COMPLETED)

    e = AnotherTinyEngine()

    @e.on(BCEvents.B_EVENT_COMPLETED)
    def check_b():
        assert e.state.b == e._allowed_events_counts[BCEvents.B_EVENT_STARTED]

    @e.on(BCEvents.C_EVENT_COMPLETED)
    def check_c():
        assert e.state.c == e._allowed_events_counts[BCEvents.C_EVENT_STARTED]

    e.run(10, 20)
    assert e.state.b == 10
    assert e.state.c == 20 * 10

    e.state.b = 3
    e.state.c = 4

    assert e._allowed_events_counts[BCEvents.B_EVENT_STARTED] == 3
    assert e._allowed_events_counts[BCEvents.C_EVENT_STARTED] == 4
    assert e._allowed_events_counts[BCEvents.B_EVENT_COMPLETED] == 3
    assert e._allowed_events_counts[BCEvents.C_EVENT_COMPLETED] == 4

    e.run(10, 20, reset=False)
    assert e.state.b == 10
    assert e.state.c == 20 * (10 - 3)

    with pytest.raises(AttributeError, match=r"can't set attribute"):
        e.state = EventsDrivenState()

    e.state.b = 4
    e.state.c = 5

    assert e._allowed_events_counts[BCEvents.B_EVENT_STARTED] == 4
    assert e._allowed_events_counts[BCEvents.C_EVENT_STARTED] == 5
    assert e._allowed_events_counts[BCEvents.B_EVENT_STARTED] == 4
    assert e._allowed_events_counts[BCEvents.C_EVENT_STARTED] == 5

    e.run(10, 20, reset=False)
    assert e.state.b == 10
    assert e.state.c == 20 * (10 - 4)
