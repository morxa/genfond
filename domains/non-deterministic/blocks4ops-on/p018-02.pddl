(define (problem blocks-018-02)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b1) (clear b11) (clear b13) (clear b3) (clear b7) (clear b8) (handempty) (on b1 b9) (on b11 b6) (on b12 b16) (on b13 b10) (on b14 b2) (on b16 b14) (on b17 b12) (on b4 b17) (on b5 b0) (on b6 b15) (on b7 b4) (on b9 b5) (ontable b0) (ontable b10) (ontable b15) (ontable b2) (ontable b3) (ontable b8))
    (:goal (on b16 b9))
)