(define (problem beam-walk-1)
    (:domain beam-walk)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects p0 - location)
    (:init (ladder-at p0) (position p0) (up))
    (:goal (and (up) (position p0)))
)