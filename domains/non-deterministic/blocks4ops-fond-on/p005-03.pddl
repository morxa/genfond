(define (problem blocks-005-03)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4)
    (:init (clear b2) (clear b3) (clear b4) (handempty) (on b0 b1) (on b3 b0) (ontable b1) (ontable b2) (ontable b4))
    (:goal (on b3 b4))
)