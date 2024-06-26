(define (problem blocks-003-3)
    (:domain blocks3ops)
    (:requirements :equality :non-deterministic :strips)
    (:objects b0 b1 b2)
    (:init (clear b1) (on b0 b2) (on b1 b0) (ontable b2))
    (:goal (and (clear b2) (on b2 b0) (on b0 b1) (ontable b1)))
)