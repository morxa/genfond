(define (problem blocks-009-1)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8)
    (:init (clear b3) (clear b6) (clear b8) (handempty) (on b0 b4) (on b2 b5) (on b5 b0) (on b6 b2) (on b7 b1) (on b8 b7) (ontable b1) (ontable b3) (ontable b4))
    (:goal (clear b2))
)