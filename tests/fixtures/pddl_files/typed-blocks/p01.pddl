(define (problem p1)
    (:domain typed-blocks)
    (:requirements :strips :typing)
    (:objects a b - block table - obj)
    (:init (on a table) (on b table))
    (:goal (on b a))
)