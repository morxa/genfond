(define (problem blocks-008-06)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5 b6 b7)
    (:init (clear b2) (clear b3) (clear b4) (clear b6) (handempty) (on b0 b5) (on b2 b1) (on b3 b7) (on b6 b0) (ontable b1) (ontable b4) (ontable b5) (ontable b7))
    (:goal (on b5 b7))
)