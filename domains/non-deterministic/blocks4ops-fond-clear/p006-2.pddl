(define (problem blocks-006-2)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5)
    (:init (clear b0) (clear b1) (handempty) (on b0 b2) (on b2 b5) (on b3 b4) (on b5 b3) (ontable b1) (ontable b4))
    (:goal (clear b5))
)