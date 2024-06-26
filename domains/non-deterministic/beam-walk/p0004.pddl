(define (problem beam-walk-4)
    (:domain beam-walk)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects p0 p1 p2 p3 - location)
    (:init (ladder-at p0) (next-bwd p1 p0) (next-bwd p2 p1) (next-bwd p3 p2) (next-fwd p0 p1) (next-fwd p1 p2) (next-fwd p2 p3) (position p0) (up))
    (:goal (and (up) (position p3)))
)