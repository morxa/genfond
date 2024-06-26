(define (problem acrobatics-2-down)
    (:domain acrobatics)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects p0 p1 - location)
    (:init (ladder-at p0) (next-bwd p1 p0) (next-fwd p0 p1) (position p0))
    (:goal (and (up) (position p1)))
)