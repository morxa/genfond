(define (problem blocks-009-04)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8)
    (:init (clear b3) (clear b4) (clear b6) (clear b7) (handempty) (on b0 b8) (on b3 b1) (on b4 b2) (on b5 b0) (on b7 b5) (ontable b1) (ontable b2) (ontable b6) (ontable b8))
    (:goal (on b1 b4))
)