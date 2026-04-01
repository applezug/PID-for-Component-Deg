"""
Physics constraint module for component-level digital twin.
"""

from .base_physics import BasePhysics
from .compressor_physics import CompressorPhysics
from .compressor_physics_strategies import CompressorPhysicsDenorm, CompressorPhysicsNorm
from .fan_physics import FanPhysicsNorm
from .turbine_physics import TurbinePhysicsNorm
from .combustor_physics import CombustorPhysicsNorm
from .igbt_physics import IGBTPhysics

__all__ = [
    'BasePhysics', 'CompressorPhysics', 'IGBTPhysics',
    'CompressorPhysicsDenorm', 'CompressorPhysicsNorm', 'FanPhysicsNorm', 'TurbinePhysicsNorm', 'CombustorPhysicsNorm',
]
