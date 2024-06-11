(define (problem blocks-004-2)
    (:domain blocks3ops)
    (:requirements :equality :non-deterministic :strips)
    (:objects b0 b1 b2 b3)
    (:init (clear b2) (on b0 b1) (on b2 b3) (on b3 b0) (ontable b1))
    (:goal (and (clear b3) (on b3 b2) (on b2 b1) (on b1 b0) (ontable b0)))
)