(define (problem acrobatics-1-down)
    (:domain acrobatics)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects p0 - location)
    (:init (ladder-at p0) (position p0))
    (:goal (and (up) (position p0)))
)