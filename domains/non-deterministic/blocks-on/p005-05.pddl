(define (problem blocks-005-05)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4)
    (:init (clear b2) (clear b4) (handempty) (on b1 b0) (on b2 b1) (on b4 b3) (ontable b0) (ontable b3))
    (:goal (on b0 b2))
)