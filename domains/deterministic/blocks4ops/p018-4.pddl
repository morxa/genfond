(define (problem blocks-018-4)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b11) (clear b14) (clear b15) (clear b9) (handempty) (on b0 b8) (on b1 b17) (on b10 b7) (on b11 b16) (on b12 b1) (on b14 b3) (on b15 b4) (on b16 b5) (on b2 b12) (on b3 b6) (on b5 b10) (on b6 b2) (on b9 b13) (ontable b13) (ontable b17) (ontable b4) (ontable b7) (ontable b8))
    (:goal (and (clear b5) (on b5 b4) (on b4 b6) (on b6 b2) (on b2 b3) (on b3 b0) (ontable b0)))
)