import threading
import time


class NoopLockController:
    def unlock(self):
        return None

    def close(self):
        return None


class GpioLockController:
    def __init__(
        self,
        chip_path: str,
        line: str | int,
        *,
        active_low: bool,
        unlock_ms: int,
        gpiod_module=None,
        direction=None,
        value=None,
        sleep=time.sleep,
    ):
        if gpiod_module is None:
            try:
                import gpiod
                from gpiod.line import Direction, Value
            except ImportError as exc:
                raise RuntimeError(
                    "LOCK_GPIO_ENABLED=1 requires Python libgpiod bindings. "
                    "Install the gpiod package on the Orange Pi."
                ) from exc
            gpiod_module = gpiod
            direction = Direction
            value = Value

        clean_line = str(line).strip()
        if not clean_line:
            raise ValueError("LOCK_GPIO_LINE is required")

        self.chip_path = chip_path
        self.line = int(clean_line) if clean_line.isdigit() else clean_line
        self.active_low = active_low
        self.unlock_ms = unlock_ms
        self._gpiod = gpiod_module
        self._direction = direction
        self._value = value
        self._sleep = sleep
        # ponytail: one process controls one relay; per-door locks if this grows.
        self._lock = threading.Lock()

    def unlock(self):
        with self._lock:
            with self._gpiod.request_lines(
                self.chip_path,
                consumer="palmgate-lock",
                config={
                    self.line: self._gpiod.LineSettings(
                        direction=self._direction.OUTPUT,
                        active_low=self.active_low,
                        output_value=self._value.INACTIVE,
                    )
                },
            ) as request:
                try:
                    request.set_value(self.line, self._value.ACTIVE)
                    self._sleep(self.unlock_ms / 1000)
                finally:
                    request.set_value(self.line, self._value.INACTIVE)

    def close(self):
        return None


def build_lock_controller():
    from app.config import (
        LOCK_ACTIVE_LOW,
        LOCK_GPIO_CHIP,
        LOCK_GPIO_ENABLED,
        LOCK_GPIO_LINE,
        LOCK_UNLOCK_MS,
    )

    if not LOCK_GPIO_ENABLED:
        return NoopLockController()

    return GpioLockController(
        LOCK_GPIO_CHIP,
        LOCK_GPIO_LINE,
        active_low=LOCK_ACTIVE_LOW,
        unlock_ms=LOCK_UNLOCK_MS,
    )
