(define (problem blocks-005-09)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4)
    (:init (clear b1) (clear b3) (clear b4) (handempty) (on b3 b2) (on b4 b0) (ontable b0) (ontable b1) (ontable b2))
    (:goal (on b2 b0))
)