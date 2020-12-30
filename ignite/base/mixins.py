import functools
import logging
import weakref
from collections import OrderedDict, defaultdict
from typing import Any, Callable, List, Optional, Union, Mapping

from ignite.engine.events import CallableEventWithFilter, EventEnum, Events, EventsList, RemovableEventHandle
from ignite.engine.utils import _check_signature


class Serializable:

    _state_dict_all_req_keys = ()  # type: tuple
    _state_dict_one_of_opt_keys = ()  # type: tuple

    def state_dict(self) -> OrderedDict:
        pass

    def load_state_dict(self, state_dict: Mapping) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Argument state_dict should be a dictionary, but given {type(state_dict)}")

        for k in self._state_dict_all_req_keys:
            if k not in state_dict:
                raise ValueError(
                    f"Required state attribute '{k}' is absent in provided state_dict '{state_dict.keys()}'"
                )
        opts = [k in state_dict for k in self._state_dict_one_of_opt_keys]
        if len(opts) > 0 and ((not any(opts)) or (all(opts))):
            raise ValueError(f"state_dict should contain only one of '{self._state_dict_one_of_opt_keys}' keys")


class EventsDriven:
    """Base class for events-driven engines without state.

    """
    def __init__(self):
        # Add auto events registering feature ?
        self._event_handlers = defaultdict(list)
        self._allowed_events = []
        self._allowed_events_counts = {}
        self.last_event_name = None
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    def register_events(
        self, *event_names: Union[List[str], List[EventEnum]], event_to_attr: Optional[Mapping] = None
    ) -> None:
        """Add events that can be fired.

        Args:
            *event_names (iterable): Defines the name of the event being supported. New events can be a str
                or an object derived from :class:`~ignite.engine.events.EventEnum`.
            event_to_attr (dict, optional): A dictionary to map an event to a state attribute.
        """
        if not (event_to_attr is None or isinstance(event_to_attr, Mapping)):
            raise ValueError("Expected event_to_attr to be dictionary. Got {}.".format(type(event_to_attr)))

        for index, e in enumerate(event_names):
            if not isinstance(e, (str, EventEnum)):
                raise TypeError(
                    "Value at {} of event_names should be a str or EventEnum, but given {}".format(index, e)
                )
            self._allowed_events.append(e)
            self._allowed_events_counts[e] = 0

    def _handler_wrapper(self, handler: Callable, event_name: Any, event_filter: Callable) -> Callable:
        # signature of the following wrapper will be inspected during registering to check if engine is necessary
        # we have to build a wrapper with relevant signature : solution is functools.wraps
        @functools.wraps(handler)
        def wrapper(*args, **kwargs) -> Any:
            # event = self.state.get_event_attrib_value(event_name)
            event = self._allowed_events_counts[event_name]
            if event_filter(self, event):
                return handler(*args, **kwargs)

        # setup input handler as parent to make has_event_handler work
        wrapper._parent = weakref.ref(handler)
        return wrapper

    def add_event_handler(self, event_name: Any, handler: Callable, *args, **kwargs):
        """Add an event handler to be executed when the specified event is fired.

        Args:
            event_name: An event or a list of events to attach the handler. Valid events are
                from :class:`~ignite.engine.events.Events` or any ``event_name`` added by
                :meth:`~ignite.base.mixins.EventsDriven.register_events`.
            handler (callable): the callable event handler that should be invoked. No restrictions on its signature.
                The first argument can be optionally `engine`, the :class:`~ignite.base.mixins.EventsDriven` object,
                handler is bound to.
            *args: optional args to be passed to ``handler``.
            **kwargs: optional keyword args to be passed to ``handler``.
        """
        if isinstance(event_name, EventsList):
            for e in event_name:
                self.add_event_handler(e, handler, *args, **kwargs)
            return RemovableEventHandle(event_name, handler, self)
        if (
            isinstance(event_name, CallableEventWithFilter)
            and event_name.filter != CallableEventWithFilter.default_event_filter
        ):
            event_filter = event_name.filter
            handler = self._handler_wrapper(handler, event_name, event_filter)

        if event_name not in self._allowed_events:
            self.logger.error("attempt to add event handler to an invalid event %s.", event_name)
            raise ValueError("Event {} is not a valid event for this {}.".format(event_name, self.__class__.__name__))

        event_args = (Exception(),) if event_name == Events.EXCEPTION_RAISED else ()
        try:
            _check_signature(handler, "handler", self, *(event_args + args), **kwargs)
            self._event_handlers[event_name].append((handler, (self,) + args, kwargs))
        except ValueError:
            _check_signature(handler, "handler", *(event_args + args), **kwargs)
            self._event_handlers[event_name].append((handler, args, kwargs))
        self.logger.debug("added handler for event %s.", event_name)

        return RemovableEventHandle(event_name, handler, self)

    @staticmethod
    def _assert_non_filtered_event(event_name: Any):
        if (
            isinstance(event_name, CallableEventWithFilter)
            and event_name.filter != CallableEventWithFilter.default_event_filter
        ):
            raise TypeError(
                "Argument event_name should not be a filtered event, please use event without any event filtering"
            )

    def has_event_handler(self, handler: Callable, event_name: Optional[Any] = None):
        """Check if the specified event has the specified handler.

        Args:
            handler (callable): the callable event handler.
            event_name: The event the handler attached to. Set this
                to ``None`` to search all events.
        """
        if event_name is not None:
            if event_name not in self._event_handlers:
                return False
            events = [event_name]
        else:
            events = self._event_handlers
        for e in events:
            for h, _, _ in self._event_handlers[e]:
                if self._compare_handlers(handler, h):
                    return True
        return False

    @staticmethod
    def _compare_handlers(user_handler: Callable, registered_handler: Callable) -> bool:
        if hasattr(registered_handler, "_parent"):
            registered_handler = registered_handler._parent()
        return registered_handler == user_handler

    def remove_event_handler(self, handler: Callable, event_name: Any):
        """Remove event handler `handler` from registered handlers of the EventsDriven instance

        Args:
            handler (callable): the callable event handler that should be removed
            event_name: The event the handler attached to.

        """
        if event_name not in self._event_handlers:
            raise ValueError("Input event name '{}' does not exist".format(event_name))

        new_event_handlers = [
            (h, args, kwargs)
            for h, args, kwargs in self._event_handlers[event_name]
            if not self._compare_handlers(handler, h)
        ]
        if len(new_event_handlers) == len(self._event_handlers[event_name]):
            raise ValueError("Input handler '{}' is not found among registered event handlers".format(handler))
        self._event_handlers[event_name] = new_event_handlers

    def on(self, event_name, *args, **kwargs):
        """Decorator shortcut for add_event_handler.

        Args:
            event_name: An event to attach the handler to. Valid events are from :class:`~ignite.engine.events.Events`
                or any ``event_name`` added by :meth:`~ignite.base.mixins.EventsDriven.register_events`.
            *args: optional args to be passed to `handler`.
            **kwargs: optional keyword args to be passed to `handler`.
        """

        def decorator(f: Callable) -> Callable:
            self.add_event_handler(event_name, f, *args, **kwargs)
            return f

        return decorator

    def _fire_event(self, event_name: Any, *event_args, **event_kwargs) -> None:
        """Execute all the handlers associated with given event.

        This method executes all handlers associated with the event
        `event_name`. Optional positional and keyword arguments can be used to
        pass arguments to **all** handlers added with this event. These
        arguments updates arguments passed using :meth:`~ignite.base.mixins.EventsDriven.add_event_handler`.

        Args:
            event_name: event for which the handlers should be executed. Valid
                events are from :class:`~ignite.engine.events.Events` or any `event_name` added by
                :meth:`~ignite.base.mixins.EventsDriven.register_events`.
            *event_args: optional args to be passed to all handlers.
            **event_kwargs: optional keyword args to be passed to all handlers.

        """
        if event_name in self._allowed_events:
            self.logger.debug("firing handlers for event %s ", event_name)
            self.last_event_name = event_name
            self._allowed_events_counts[event_name] += 1
            for func, args, kwargs in self._event_handlers[event_name]:
                kwargs.update(event_kwargs)
                first, others = ((args[0],), args[1:]) if (args and args[0] == self) else ((), args)
                func(*first, *(event_args + others), **kwargs)

    def fire_event(self, event_name: Any) -> None:
        """Execute all the handlers associated with given event.

        Args:
            event_name: event for which the handlers should be executed. Valid
                events are from :class:`~ignite.engine.events.Events` or any `event_name` added by
                :meth:`~ignite.base.mixins.EventsDriven.register_events`.
        """
        return self._fire_event(event_name)

    def _reset_allowed_events_counts(self):
        for k in self._allowed_events_counts:
            self._allowed_events_counts[k] = 0


class EventsDrivenState:
    """State for EventsDriven class. State attributed are automatically synchronized with
    EventsDriven counters.
    """

    def __init__(
        self, engine: Optional[EventsDriven] = None, event_to_attr: Optional[Mapping[Any, str]] = None, **kwargs: Any
    ):
        if event_to_attr is not None and engine is None:
            raise ValueError("Both engine and event_to_attr should be provided, but only event_to_attr is given")

        self.event_to_attr = event_to_attr  # type: Optional[Mapping[str, str]]
        self.engine = engine  # type: Optional[EventsDriven]

        self._attr_to_event = None
        if event_to_attr is not None:
            # Create inverse mapping
            self._attr_to_event = defaultdict(list)
            for k, v in event_to_attr.items():
                self._attr_to_event[v].append(k)

    def __getattr__(self, attr):
        evnts = None
        if self._attr_to_event and attr in self._attr_to_event:
            evnts = self._attr_to_event[attr]

        if self.engine and evnts:
            # return max of available event counts
            counts = [self.engine._allowed_events_counts[e] for e in evnts]
            return max(counts)

        raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, attr))

    def __setattr__(self, attr, value):
        if all([a in self.__dict__ for a in ["engine", "_attr_to_event"]]) and self.__dict__["engine"]:
            self__attr_to_event = self.__dict__["_attr_to_event"]
            evnts = None
            if self__attr_to_event and attr in self__attr_to_event:
                evnts = self__attr_to_event[attr]
            self_engine = self.__dict__["engine"]
            if self_engine and evnts:
                # Set all counters to provided value
                for e in evnts:
                    if e in self_engine._allowed_events_counts:
                        self_engine._allowed_events_counts[e] = value
                return

        super().__setattr__(attr, value)


class EventsDrivenWithState(EventsDriven):
    """Base class for events-driven engines with state.
    """

    def __init__(self):
        super(EventsDrivenWithState, self).__init__()
        self._state = EventsDrivenState(self)

    @property
    def state(self) -> EventsDrivenState:
        return self._state

    @state.setter
    def state(self, new_state: EventsDrivenState):
        raise AttributeError("can't set attribute")
