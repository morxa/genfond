(define (problem blocks-002-1)
    (:domain blocks3ops)
    (:requirements :equality :non-deterministic :strips)
    (:objects b0 b1)
    (:init (clear b1) (on b1 b0) (ontable b0))
    (:goal (and (clear b1) (on b1 b0) (ontable b0)))
)