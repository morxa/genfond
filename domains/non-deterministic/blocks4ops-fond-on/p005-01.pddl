(define (problem blocks-005-01)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4)
    (:init (clear b1) (clear b4) (handempty) (on b2 b0) (on b3 b2) (on b4 b3) (ontable b0) (ontable b1))
    (:goal (on b0 b2))
)