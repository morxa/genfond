(define (problem blocks-007-2)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5 b6)
    (:init (clear b2) (clear b4) (clear b5) (handempty) (on b0 b3) (on b3 b6) (on b5 b0) (on b6 b1) (ontable b1) (ontable b2) (ontable b4))
    (:goal (clear b6))
)